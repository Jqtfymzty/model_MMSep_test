import os
os.environ["ARK_API_KEY"] = "YOUR_ARK_API_KEY"
import base64
import argparse
import json
from volcenginesdkarkruntime import Ark
from tqdm import tqdm
import math

NUM_SECONDS_TO_SLEEP = 3
SYSTEM_PROMPT = "You are a helpful and precise assistant for checking the quality of the answers."

REL_INSTRUCTION = """
We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above. 
The user asks the question on observing an image. For your reference, the visual content in the image is represented with a few sentences describing the image. \n
Please rate the helpfulness, relevance, accuracy, level of details of their responses. 
Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\n
Please first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space.\n
In the subsequent line, please provide a comprehensive explanation of your evaluation.
"""

ABS_INSTRUCTION = """
We would like to request your feedback on the performance of an AI assistant in response to the user question displayed above. 
The user asks the question on observing an image. For your reference, the visual content in the image is represented with a few sentences describing the image. \n
Please rate the helpfulness, relevance, accuracy, level of details of the response. 
The assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\n
Please first output a single line containing only one value indicating the score for the model answer.\n
In the subsequent line, please provide a comprehensive explanation of your evaluation."""

IMG_INSTRUCTION = """
We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above. 
The user asks the question on observing an image.\n
Please rate the helpfulness, relevance, accuracy, level of details of their responses. 
Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\n
Please first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space.\n
In the subsequent line, please provide a comprehensive explanation of your evaluation, 
avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.
"""

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def load_reference_and_answer(reference_file, answer_file, num_chunks, chunk_idx):
    reference = [json.loads(r) for r in open(os.path.expanduser(reference_file), "r")]
    answers = [json.loads(a) for a in open(os.path.expanduser(answer_file), "r")]
    reference = get_chunk(reference, num_chunks, chunk_idx)
    answers = get_chunk(answers, num_chunks, chunk_idx)
    
    return reference, answers


def doubao_prompt_txt(question, ref_answer, model_answer, mode="rel"):
    if mode == "rel":
        prompt = f"""
        [Question]\n{question}\n\n
        [Assistant 1]\n{ref_answer}\n\n[End of Assistant 1]\n\n
        [Assistant 2]\n{model_answer}\n\n[End of Assistant 2]\n\n
        [System]\n{REL_INSTRUCTION}
        """
    else:  # abs
        raise NotImplementedError("Absolute evaluation is not implemented yet.")
    
    return prompt


def doubao_chat_step_txt(client, prompt: str, max_tokens: int = 256, temperature: float = 0.0, top_p: float = 1.0):
    try:
        response = client.chat.completions.create(
            # use your endpoint id
            model="endpoint_id",
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            thinking={"type": "disabled"},
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        return response.choices[0].message.content
    except Exception as e:
        print(f"Error during doubao chat step: {e}")
        return "-1 -1\nError during evaluation."


def doubao_prompt_img(question, ref_answer, model_answer, image_str, mode="rel"):
    if mode == "rel":
        prompt = [{
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_str}",
                }
            },{
                "type": "text",
                "text": f"""
                        [Question]\n{question}\n\n
                        [Assistant 1]\n{ref_answer}\n\n[End of Assistant 1]\n\n
                        [Assistant 2]\n{model_answer}\n\n[End of Assistant 2]\n\n
                        [System]\n{IMG_INSTRUCTION}
                        """,
            }]
    else:  # abs
        raise NotImplementedError("Absolute evaluation is not implemented yet.")
    
    return prompt

def doubao_chat_step_img(client, prompt: str, max_tokens: int = 256, temperature: float = 0.0, top_p: float = 1.0):
    try:
        response = client.chat.completions.create(
            # use your endpoint id
            model="endpoint_id",
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            thinking={"type": "disabled"},
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        return response.choices[0].message.content
    except Exception as e:
        print(f"Error during doubao chat step: {e}")
        return "-1 -1\nError during evaluation."


def parse_score(response: str, mode="rel"):
    try:
        lines = response.strip().split("\n")
        if mode == "rel":
            score_line = lines[0].strip()
            explanation = "\n".join(lines[1:]).strip()
            # 将可能是浮点的数值转换为整数
            scores = list(map(lambda x: int(float(x)), score_line.split()))
            assert len(scores) == 2, "There should be two scores for relative evaluation."
        else:  # abs
            score_line = lines[0].strip()
            explanation = "\n".join(lines[1:]).strip()
            scores = [int(score_line)]

    except Exception as e:
        print(f"Error parsing score from response: {e}")
        if mode == "rel":
            scores = [-1, -1]
        else:
            scores = [-1]
        explanation = "Error parsing score."
    
    return scores, explanation


def eval_doubao_review(args):
    if not os.path.exists(args.reference):
        raise FileNotFoundError(f"Reference file {args.reference} does not exist.")
    if not os.path.exists(args.answer):
        raise FileNotFoundError(f"Answer file {args.answer} does not exist.")
    
    # load reference and answer
    references, answers = load_reference_and_answer(
        args.reference, args.answer, args.num_chunks, args.chunk_idx)
    assert len(references) == len(answers), "Reference and answer files must have the same number of entries."

    # result
    save_file = os.path.expanduser(args.save_file)
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    sf = open(save_file, "w")

    # client and tokenizer
    client = Ark(
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        api_key=os.environ.get("ARK_API_KEY"),
    )

    for ref, ans in tqdm(zip(references, answers), total=len(references), ncols=100):
        # prompt
        question = ref["question"] if "question" in ref else ref["prompt"]
        ref_answer = ref["answer"] if "answer" in ref else ref["text"]
        model_answer = ans["text"]

        # response from doubao
        if args.scorer == "doubao_txt":
            prompt = doubao_prompt_txt(question, ref_answer, model_answer, args.mode)
            response = doubao_chat_step_txt(client, prompt, args.max_tokens, args.temperature, args.top_p)

        elif args.scorer == "doubao_img":
            image_path = os.path.join(args.image_folder, ref["question_id"] + ".jpg")
            with open(image_path, "rb") as img_f:
                image_bytes = img_f.read()
            image_str = base64.b64encode(image_bytes).decode("utf-8")
            prompt = doubao_prompt_img(question, ref_answer, model_answer, image_str, args.mode)
            response = doubao_chat_step_img(client, prompt, args.max_tokens, args.temperature, args.top_p)

        else:
            raise NotImplementedError(f"Scorer {args.scorer} is not implemented yet.")
        
        # parse score
        scores, explanation = parse_score(response, args.mode)

        # save result
        result = {
            "question_id": ref["question_id"],
            "image": ref["question_id"] + ".jpg",
            "question": question,
            "ref_score": scores[0] if args.mode == "rel" else None,
            "model_score": scores[1] if args.mode == "rel" else scores[0],
            "ref": ref_answer,
            "model_ans": model_answer,
            "explanation": explanation,
        }
        sf.write(json.dumps(result) + "\n")
    
    sf.close()
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # load reference answer and model answer
    parser.add_argument("--reference", type=str, default="./debug/debug_detail_1k.jsonl")
    parser.add_argument("--answer", type=str, required=True)
    parser.add_argument("--save-file", type=str, required=True)
    parser.add_argument("--image-folder", type=str, default="/data1/mmj/Datasets/coco2017/train2017")
    parser.add_argument("--scorer", type=str, default="doubao_txt", choices=["doubao_txt", "doubao_img", "llava", "gpt"])
    parser.add_argument("--mode", type=str, default="rel", choices=["rel", "abs"])
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)

    args = parser.parse_args()

    eval_doubao_review(args)
