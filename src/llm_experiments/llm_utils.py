import argparse
import os
import openai
import anthropic


def get_clients(model):
    openai_client = None
    anthropic_client = None

    if model.startswith("gpt-") or model.startswith("text-"):
        openai_client = openai.OpenAI()
        print(f"Using OpenAI model: {model}")
    elif model.startswith("claude-"):
        anthropic_client = anthropic.Anthropic()
        print(f"Using Anthropic model: {model}")
    else:
        raise ValueError(f"Unknown model type: {model}. Please use a model name that starts with 'gpt-', 'text-', or 'claude-'")
    return openai_client, anthropic_client


def label_text(prompt, text, client, model, temp=0.25, top_p=0.9, max_new_tokens=512, sample=False):
    if text is None:
        return None
    
    if model.startswith("gpt-") or model.startswith("text-"):
        return label_text_openai(prompt, text, client, model, temp=0.25, top_p=0.9, max_new_tokens=512, sample=False)
    elif model.startswith("claude-"):
        return label_text_anthropic(prompt, text, client, model, temp=0.25, top_p=0.9, max_new_tokens=512, sample=False)
    else:
        return "Error: Unknown model type"
    

def label_text_openai(prompt, text, client, model, temp=0.25, top_p=0.9, max_new_tokens=512, sample=False):
    try:
        full_prompt = prompt + text
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": full_prompt}],
            temperature=temp if sample else 0.0,
            top_p=top_p if sample else 1.0,
            max_tokens=max_new_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error with OpenAI API: {e}")
        return f"Error: {str(e)}"

def label_text_anthropic(prompt, text, client, model, temp=0.25, top_p=0.9, max_new_tokens=512, sample=False):
    try:
        full_prompt = prompt + text
        response = client.messages.create(
            model=model,
            max_tokens=max_new_tokens,
            temperature=temp if sample else 0.0,
            top_p=top_p if sample else 1.0,
            messages=[{"role": "user", "content": full_prompt}]
        )
        return response.content[0].text
    except Exception as e:
        print(f"Error with Anthropic API: {e}")
        return f"Error: {str(e)}"