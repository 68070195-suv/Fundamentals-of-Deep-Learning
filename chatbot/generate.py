import torch
from tokenizers import Tokenizer

from config import (
    DEVICE,
    BLOCK_SIZE,
    TOKENIZER_PATH,
    CHECKPOINT_DIR,
    PAD_TOKEN_ID,
    UNK_TOKEN_ID,
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    SYSTEM_TOKEN_ID,
    USER_TOKEN_ID,
    ASSISTANT_TOKEN_ID
)

from model.gpt import GPTModel


tokenizer = Tokenizer.from_file(
    str(TOKENIZER_PATH)
)

model = GPTModel().to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT_DIR / "best_sft.pt",
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


def build_input(prompt, history=None):

    prompt_ids = tokenizer.encode(
        prompt,
        add_special_tokens=False
    ).ids

    max_prompt_tokens = BLOCK_SIZE - 3

    if len(prompt_ids) > max_prompt_tokens:
        prompt_ids = prompt_ids[-max_prompt_tokens:]

    current_ids = [
        USER_TOKEN_ID,
        *prompt_ids,
        ASSISTANT_TOKEN_ID
    ]

    # History Conversation
    history_budget = BLOCK_SIZE - 1 - len(current_ids)

    history_messages = []

    if history and history_budget > 0:

        encoded_history = []

        for message in history:

            role = message["role"]
            content = message["content"]

            if role == "user":
                role_id = USER_TOKEN_ID

            elif role == "assistant":
                role_id = ASSISTANT_TOKEN_ID

            else:
                continue

            content_ids = tokenizer.encode(
                content,
                add_special_tokens=False
            ).ids

            message_ids = [role_id, *content_ids]

            if role == "assistant":
                message_ids.append(EOS_TOKEN_ID)

            encoded_history.append(message_ids)


        selected = []

        for message_ids in reversed(encoded_history):

            if len(message_ids) <= history_budget:
                selected.append(message_ids)
                history_budget -= len(message_ids)
            else:
                break

        history_messages = list(reversed(selected))

    token_ids = [BOS_TOKEN_ID]

    for message_ids in history_messages:
        token_ids.extend(message_ids)

    token_ids.extend(current_ids)

    return torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=DEVICE
    )


def apply_repetition_penalty(
    logits,
    input_ids,
    repetition_penalty
):

    if repetition_penalty == 1.0:
        return logits

    if repetition_penalty < 1.0:
        raise ValueError(
            "repetition_penalty must be >= 1.0"
        )


    previous_tokens = torch.unique(input_ids)

    logits = logits.clone()

    previous_logits = logits[0, previous_tokens]

    logits[0, previous_tokens] = torch.where(
        previous_logits < 0,
        previous_logits * repetition_penalty,
        previous_logits / repetition_penalty
    )

    return logits


def sample_next_token(
    logits,
    temperature,
    top_k,
    top_p
):

    logits = logits / temperature

    # Top-K
    if top_k > 0:

        k = min(top_k, logits.size(-1))

        values, indices = torch.topk(
            logits,
            k=k,
            dim=-1
        )

        filtered_logits = torch.full_like(
            logits,
            float("-inf")
        )

        filtered_logits.scatter_(
            1,
            indices,
            values
        )

        logits = filtered_logits

    # Top-P
    if 0 < top_p < 1:

        sorted_logits, sorted_indices = torch.sort(
            logits,
            descending=True,
            dim=-1
        )

        sorted_probs = torch.softmax(
            sorted_logits,
            dim=-1
        )

        cumulative_probs = torch.cumsum(
            sorted_probs,
            dim=-1
        )

        sorted_indices_to_remove = (
            cumulative_probs > top_p
        )

        sorted_indices_to_remove[..., 1:] = (
            sorted_indices_to_remove[..., :-1].clone()
        )

        sorted_indices_to_remove[..., 0] = False

        indices_to_remove = torch.zeros_like(
            logits,
            dtype=torch.bool
        )

        indices_to_remove.scatter_(
            1,
            sorted_indices,
            sorted_indices_to_remove
        )

        logits = logits.masked_fill(
            indices_to_remove,
            float("-inf")
        )

    probabilities = torch.softmax(
        logits,
        dim=-1
    )

    next_token = torch.multinomial(
        probabilities,
        num_samples=1
    )

    return next_token


@torch.no_grad()
def generate(
    prompt,
    history=None,
    max_new_tokens=100,
    temperature=0.7,
    top_k=20,
    top_p=0.9,
    repetition_penalty=1.15
):

    if not prompt.strip():
        raise ValueError("Prompt must not be empty")

    if temperature <= 0:
        raise ValueError("Temperature must be greater than 0")

    if top_k < 0:
        raise ValueError("top_k must be >= 0")

    if not 0 < top_p <= 1:
        raise ValueError("top_p must be between 0 and 1")

    if repetition_penalty < 1.0:
        raise ValueError("repetition_penalty must be >= 1.0")

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be greater than 0")

    input_ids = build_input(
        prompt,
        history
    )

    prompt_length = input_ids.size(1)

    forbidden_ids = [
        PAD_TOKEN_ID,
        UNK_TOKEN_ID,
        BOS_TOKEN_ID,
        SYSTEM_TOKEN_ID,
        USER_TOKEN_ID,
        ASSISTANT_TOKEN_ID
    ]

    forbidden_ids = [
        token_id
        for token_id in forbidden_ids
        if 0 <= token_id < model.lm_head.out_features
    ]

    for _ in range(max_new_tokens):

        context = input_ids[:, -BLOCK_SIZE:]

        logits = model(context)

        logits = logits[:, -1, :]

        logits[:, forbidden_ids] = float("-inf")

        logits = apply_repetition_penalty(
            logits,
            context,
            repetition_penalty
        )

        next_token = sample_next_token(
            logits,
            temperature,
            top_k,
            top_p
        )

        input_ids = torch.cat(
            [input_ids, next_token],
            dim=1
        )

        if next_token.item() == EOS_TOKEN_ID:
            break

    generated_ids = input_ids[
        0,
        prompt_length:
    ].tolist()

    if EOS_TOKEN_ID in generated_ids:
        generated_ids = generated_ids[
            :generated_ids.index(EOS_TOKEN_ID)
        ]

    response = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    )

    return response.strip()


if __name__ == "__main__":

    history = []

    while True:

        prompt = input("\nUser : ")
        print("\nUser : " + prompt)

        if prompt.lower() in ["exit", "quit"]:
            break

        if prompt.lower() == "reset":
            history.clear()
            print("\nBOT : เริ่มบทสนทนาใหม่แล้วครับ")
            continue

        if not prompt.strip():
            continue

        try:

            response = generate(
                prompt,
                history=history,
                temperature=0.7,
                top_k=20,
                top_p=0.9,
                repetition_penalty=1.15,
                max_new_tokens=100
            )

            print("\nBOT :", response)

            history.append({
                "role": "user",
                "content": prompt
            })

            history.append({
                "role": "assistant",
                "content": response
            })

        except Exception as error:

            print("\nเกิดข้อผิดพลาด:", error)
