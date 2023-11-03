import contextlib
import functools
import json
import openai
import time
import os


def load_pricing():
    with open('pricing.json') as f:
        return json.load(f)

PRICING = load_pricing()

def get_rate_per_token(model, prompt_tokens, completion_tokens):
    if model == "text-embedding-ada-002":
        usage = PRICING[model]["embedding"]
        tokens = prompt_tokens + completion_tokens
        cost = (usage / 1000) * tokens
        return cost
    elif model in PRICING.keys():
        prompt = PRICING[model]["prompt"]
        cost_prompt = (prompt / 1000) * prompt_tokens
        completion = PRICING[model]["completion"]
        cost_completion = (completion / 1000) * completion_tokens
        cost = cost_prompt + cost_completion
        return cost
    else:
        raise ValueError("Model pricing unknown")


def log_cost_and_timestamp(func, asynch=False, websocket=None):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        model = kwargs.get("model", None)
        if model is None:
            model = kwargs.get("engine")

        if asynch:
            response = await func(*args, **kwargs)
        else:
            response = func(*args, **kwargs)

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cost = get_rate_per_token(model, response["usage"]["prompt_tokens"], response["usage"]['completion_tokens'])
        cost_float = float(cost)

        if websocket:
            await websocket.send_json({'sender': 'cost_calculator', 'message': cost_float}, mode='text')

        cost_str = f"{cost_float:.10f}"
        cost_formatted = cost_str.rstrip('0').rstrip('.')

        log_file = 'api_calls.log'
        if not os.path.exists(log_file):
            with open(log_file, 'w') as f:
                f.write('')
        with open(log_file, 'r') as f:
            total_cost = 0
            for line in f:
                entry = json.loads(line)
                total_cost += float(entry['cost'])

        total_cost += cost_float
        total_cost_str = f"{total_cost:.10f}"
        total_cost_formatted = total_cost_str.rstrip('0').rstrip('.')

        with open(log_file, 'a') as f:
            log_entry = {
                'timestamp': timestamp,
                'model': model,
                'cost': cost_formatted,
                'total_cost': total_cost_formatted,
            }
            f.write(json.dumps(log_entry) + '\n')
        return response

    return wrapper


@contextlib.contextmanager
def openai_api_listener(websocket):
    original_create = openai.Completion.create
    openai.Completion.create = log_cost_and_timestamp(original_create, websocket=websocket)
    original_chat_create = openai.ChatCompletion.create
    openai.ChatCompletion.create = log_cost_and_timestamp(original_chat_create, websocket=websocket)
    original_acreate = openai.Completion.acreate
    openai.Completion.acreate = log_cost_and_timestamp(original_acreate, asynch=True, websocket=websocket)
    original_chat_acreate = openai.ChatCompletion.acreate
    openai.ChatCompletion.acreate = log_cost_and_timestamp(original_chat_acreate, asynch=True, websocket=websocket)
    try:
        yield
    finally:
        openai.Completion.create = original_create
        openai.ChatCompletion.create = original_chat_create
        openai.Completion.acreate = original_acreate
        openai.ChatCompletion.acreate = original_chat_acreate
