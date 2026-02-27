import pandas as pd
from string import Template
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import os
import re
from pygments.lexers import guess_lexer
from pygments.util import ClassNotFound
import ast
from tqdm import tqdm
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_script_dir = os.path.dirname(os.path.abspath(__file__))


def _resolve_latest_checkpoint(base_dir: str) -> str:
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"LoRA 输出目录不存在: {base_dir}")

    candidates = []
    for name in os.listdir(base_dir):
        path = os.path.join(base_dir, name)
        if os.path.isdir(path) and name.startswith("checkpoint-"):
            suffix = name.split("checkpoint-")[-1]
            if suffix.isdigit():
                candidates.append((int(suffix), path))
    if not candidates:
        raise FileNotFoundError(f"未找到 checkpoint 目录: {base_dir}/checkpoint-*")
    return max(candidates, key=lambda item: item[0])[1]


# 第一步：自动选择最新 LoRA checkpoint（本地路径，避免误走 HuggingFace Hub）
peft_output_dir = os.path.normpath(os.path.abspath(os.path.join(_script_dir, "..", "output-src-qw")))
peft_model_path = _resolve_latest_checkpoint(peft_output_dir)
config = PeftConfig.from_pretrained(peft_model_path)

# 第二步：加载基础模型（与 train.py 保持一致，使用本地 models 目录）
model_id = os.path.normpath(os.path.abspath(os.path.join(_script_dir, "..", "models")))
if not os.path.isdir(model_id):
    raise FileNotFoundError(f"基础模型目录不存在: {model_id}")
# 第二步：加载基础模型（LoRA 是在这个模型上微调的）
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,  # 原始模型
    trust_remote_code=True,
    # device_map="auto",
    device_map={"": device},
    torch_dtype="auto"
)

# 第三步：加载 LoRA 适配器权重
# 消融
# p_model=base_model
p_model = PeftModel.from_pretrained(base_model, peft_model_path).to(device)
p_model.eval()

# 第四步：加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

def is_probably_python(text: str) -> bool:
    try:
        lexer = guess_lexer(text)
        return 'Python' in lexer.name
    except ClassNotFound:
        return False

def extract_code(text: list):
    pattern = fr"```python\s*(.*?)```"
    code_blocks = re.findall(pattern, text, re.DOTALL)
    if code_blocks:
        return code_blocks[0].strip()
    # =================================
    pattern = fr"```\s*(.*?)```"
    code_blocks = re.findall(pattern, text, re.DOTALL)
    if code_blocks:
        return code_blocks[0].strip()
    # =================================
    text.strip()
    possible_codes = text.split("```")
    # =================================
    if is_probably_python(possible_codes[0]):
        return possible_codes[0].strip()
    elif is_probably_python(possible_codes[-1]):
        return possible_codes[-1].strip()

    return text


# 读取数据（显式文件路径，避免 pandas 将字符串当作文本解析；优先项目根，否则 eval/to_be_eval）
_test_jsonl = os.path.normpath(os.path.join(_script_dir, "..", "test-sug.jsonl"))
if not os.path.isfile(_test_jsonl):
    _test_jsonl = os.path.normpath(os.path.join(_script_dir, "..", "eval", "to_be_eval", "test-sug.jsonl"))
if not os.path.isfile(_test_jsonl):
    raise FileNotFoundError(f"未找到 test-sug.jsonl，请放在项目根或 eval/to_be_eval/")
with open(_test_jsonl, "r", encoding="utf-8") as f:
    df = pd.read_json(f, lines=True)
print(df.shape)

input_template = Template("Input code:\n$input\nSuggestions:\n$suggestions\nOutput: ```python\n {{optimized code}} \n``` <|EOS|>")
system="""You are an AI programming assistant. Try your best to optimize the Input code with focusing on reducing its runtime while maintaining correctness.
Feel free to refer to the suggestions below for potential optimizations, but you're not restricted to them.
Applicability represents the degree to which a suggestion fits the input code.
Rate represents the degree to which a suggestion improves the input code."""

tqdm.pandas()

torch.cuda.empty_cache()

def infer(row):
    input = row['slow_code_col']
    # suggestions = ast.literal_eval(row['suggestion'])
    suggestions = row['suggestion']
    suggestions = "\n".join(
        [f"{index + 1}. Applicability:{suggestion['distance']} Rate:{suggestion['rate']}\n{suggestion['text']}" for
         index, suggestion in enumerate(suggestions)])
    prompt = input_template.substitute(input=input, suggestions=suggestions)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt}
    ]

    model_inputs = tokenizer.apply_chat_template(
        messages,
        # tokenize=False,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(p_model.device)
    with torch.inference_mode():
        outputs = p_model.generate(
            model_inputs,
            max_new_tokens=512,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1,
        )
    response = tokenizer.decode(outputs[0][len(model_inputs[0]):], skip_special_tokens=True)
    del model_inputs
    del outputs
    torch.cuda.empty_cache()

    code = extract_code(response)
    return code

df['model_generated_potentially_faster_code_col'] = df.progress_apply(infer, axis=1)
torch.cuda.empty_cache()
df.to_json("test-qw2.5-7b-i-fp.jsonl",lines=True,orient='records')
empty_rows = df[df['model_generated_potentially_faster_code_col'] == '']
print(empty_rows.shape)