import os
from enum import Enum
from os import environ
from typing import *
from typing import Tuple

import google.generativeai as genai
from google.generativeai.protos import FunctionCall as Gemini_FunctionCall, Content, Part
from google.generativeai.types import SafetySettingDict, HarmBlockThreshold, HarmCategory
from openai import OpenAI
from gradio_client import Client

"""
LLMs are defined as instances of the LLM_API class
LLM negotators are different classes which use the LLM_API
"""

from abc import ABC, abstractmethod

from llama_cpp import Llama

class LLM_API(ABC):
  class Role(Enum):
    SYSTEM = 0
    ASSISTANT = 1
    USER = 2

  def __init__(self, name: str, system_prompt: str, api_token_env_var: str, model_name: str,
               instance_name: Optional[str] = None):
    self.name = name
    self.system_prompt = system_prompt
    #self.token = environ[api_token_env_var]
    self.model_name = model_name
    #self.chat_history: List[Dict[str, Any]] = []
    self.instance_name = instance_name if instance_name is not None else model_name
    self.chat_history = []
    if api_token_env_var:
          self.token = environ[api_token_env_var]
    else:
          self.token = None
  """
  Query the LLM
  Returns a tuple of (response, stop_condition, ....)
  """

  @abstractmethod
  def query(self, prompt: str) -> Union[str, Any]:
    raise NotImplementedError

  """
  Add a message to the LLM message history
  """

  @abstractmethod
  def add_text_to_history(self, role: Role, text: str) -> None:
    raise NotImplementedError

  """
  Convert a Role enum to a string
  """

  @abstractmethod
  def role_to_str(self, role: Role) -> str:
    raise NotImplementedError

  def clear_history(self):
    self.chat_history = []

  def get_last_message_text(self) -> str:
    return self._history_element_text(self.chat_history[-1])

  def history_to_text(self, include_roles: bool) -> List[str]:
    ss = []

    for element in self.chat_history:
      s = ""
      if include_roles:
        s += self._history_element_role(element)
        s += ": "
      s += self._history_element_text(element)
      ss.append(s)

    return ss

  @abstractmethod
  def _history_element_text(self, element: Any) -> str:
    raise NotImplementedError

  @abstractmethod
  def _history_element_role(self, element: Any) -> str:
    raise NotImplementedError


class Gemini(LLM_API):
  def __init__(self,
               system_prompt: str,
               model_name: str,
               api_token_env_var: str = "GOOGLE_API_KEY",
               max_output_tokens: int = 300,
               stop_sequences: Optional[List[str]] = None,
               functions: Optional[List[Callable]] = None,
               temperature: float = 1.0,
               top_p: float = 0.95,
               top_k: int = 64,
               response_mime_type: str = "text/plain",
               instance_name: Optional[str] = None
               ):
    super().__init__("Gemini", system_prompt, api_token_env_var, model_name, instance_name)
    self.generation_config = {
      "temperature":        temperature,
      "top_p":              top_p,
      "top_k":              top_k,
      "max_output_tokens":  max_output_tokens,
      "response_mime_type": response_mime_type,
    }
    if stop_sequences is not None:
      self.generation_config["stop_sequences"] = stop_sequences

    # For some reason it keeps thinking it is bullying someone
    safety_settings: SafetySettingDict = {
      HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
      HarmCategory.HARM_CATEGORY_HARASSMENT:        HarmBlockThreshold.BLOCK_NONE,
      HarmCategory.HARM_CATEGORY_HATE_SPEECH:       HarmBlockThreshold.BLOCK_NONE,
      HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    }

    self.model = genai.GenerativeModel(
        model_name = self.model_name,
        generation_config = self.generation_config,
        system_instruction = self.system_prompt,
        tools = functions,
        safety_settings = safety_settings
    )

  def query(self, prompt: str) -> Tuple[str, List[Gemini_FunctionCall]]:
    chat = self.model.start_chat(history = self.chat_history)
    response = chat.send_message(prompt)

    # print(response)

    self.chat_history = chat.history

    text_res = None
    fc_res_list = []

    for part in response.parts:
      if (fc := part.function_call):
        fc_res_list.append(fc)
      elif (t := part.text):
        text_res = t.strip()

    return (text_res, fc_res_list)

    # first_part = response.parts[0]

    # print(response)
    # print(first_part)

    # if(fc := first_part.function_call):
    #   return fc
    # elif(t := first_part.text):
    #   return t.strip()
    # else:
    #   return None

  def add_text_to_history(self, role: LLM_API.Role, text: str):
    content = Content()
    content.role = self.role_to_str(role)
    part = Part()
    part.text = text
    content.parts = [part]
    self.chat_history.append(
        content
    )

  def role_to_str(self, role: LLM_API.Role) -> str:
    assert isinstance(role, LLM_API.Role)
    if (role == LLM_API.Role.SYSTEM):
      raise AttributeError(name = "SYSTEM cannot be used for Gemini models")
    return ["system", "model", "user"][role.value]

  def _history_element_text(self, element: Any) -> str:
    return element.parts[0].text.strip()

  def _history_element_role(self, element: Any) -> str:
    return element.role.strip()


class GPT(LLM_API):
  def __init__(self,
               system_prompt: str,
               model_name: str,
               api_token_env_var: str = "OPENAI_API_KEY",
               max_output_tokens: int = 300,
               stop_sequences: Optional[List[str]] = None,
               functions: Optional[List[Dict[str, Any]]] = None,
               temperature: float = 1.0,
               top_p: float = 0.95,
               frequency_penalty: float = 0.,
               presence_penalty: float = 0.,
               instance_name: Optional[str] = None
               ):
    super().__init__("GPT", system_prompt, api_token_env_var, model_name, instance_name)
    self.generation_config = {
      "temperature":       temperature,
      "top_p":             top_p,
      "frequency_penalty": frequency_penalty,
      "presence_penalty":  presence_penalty,
      "max_tokens":        max_output_tokens,
      "tools":             functions
    }
    if stop_sequences is not None:
      self.generation_config["stop"] = stop_sequences

    self.add_text_to_history(LLM_API.Role.SYSTEM, system_prompt)

    self.model = OpenAI()

  def query(self, prompt: str) -> Tuple[str, Any]:
    self.add_text_to_history(LLM_API.Role.USER, prompt)

    response = self.model.chat.completions.create(
        model = self.model_name,
        messages = self.chat_history,
        **self.generation_config
    )

    choice = response.choices[0]

    if choice.finish_reason == 'tool_calls':
      raise NotImplementedError
    else:
      res_text: str = choice.message.content
      self.add_text_to_history(LLM_API.Role.ASSISTANT, res_text)
      return res_text.strip(), []

  def add_text_to_history(self, role: LLM_API.Role, text: str):
    self.chat_history.append(
        {
          "role":    self.role_to_str(role),
          "content": [
            {
              "type": "text",
              "text": text
            }
          ]
        }
    )

  def role_to_str(self, role: LLM_API.Role) -> str:
    assert isinstance(role, LLM_API.Role)
    return ["system", "assistant", "user"][role.value]

  def clear_history(self):
    super().clear_history()
    self.add_text_to_history(LLM_API.Role.SYSTEM, self.system_prompt)

  def _history_element_text(self, element: Any) -> str:
    return element["content"][0]["text"].strip()

  def _history_element_role(self, element: Any) -> str:
    return element["role"].strip()


class DeepInfra(LLM_API):
  def __init__(self,
               system_prompt: str,
               model_name: str,
               api_token_env_var: str = "DEEPINFRA_API_KEY",
               max_output_tokens: int = 300,
               stop_sequences: Optional[List[str]] = None,
               functions: Optional[List[Dict[str, Any]]] = None,
               temperature: float = 1.0,
               top_p: float = 0.95,
               frequency_penalty: float = 0.,
               presence_penalty: float = 0.,
               instance_name: Optional[str] = None
               ):

    super().__init__("DeepInfra", system_prompt, api_token_env_var, model_name, instance_name)
    self.generation_config = {
      "stream":            False,
      "temperature":       temperature,
      "top_p":             top_p,
      "frequency_penalty": frequency_penalty,
      "presence_penalty":  presence_penalty,
      "max_tokens":        max_output_tokens,
    }
    if stop_sequences is not None:
      self.generation_config["stop"] = stop_sequences

    self.add_text_to_history(LLM_API.Role.SYSTEM, system_prompt)

    self.model = OpenAI(
        api_key = self.token,
        base_url = "https://api.deepinfra.com/v1/openai",
    )

  def query(self, prompt: str) -> Tuple[str, Any]:
    self.add_text_to_history(LLM_API.Role.USER, prompt)

    response = self.model.chat.completions.create(
        model = self.model_name,
        messages = self.chat_history,
        **self.generation_config
    )

    choice = response.choices[0]

    if choice.finish_reason == 'tool_calls':
      raise NotImplementedError
    else:
      res_text: str = choice.message.content
      self.add_text_to_history(LLM_API.Role.ASSISTANT, res_text)
      return res_text.strip(), []

  def add_text_to_history(self, role: LLM_API.Role, text: str):
    self.chat_history.append(
        {
          "role":    self.role_to_str(role),
          "content": text
        }
    )

  def role_to_str(self, role: LLM_API.Role) -> str:
    assert isinstance(role, LLM_API.Role)
    return ["system", "assistant", "user"][role.value]

  def clear_history(self):
    super().clear_history()
    self.add_text_to_history(LLM_API.Role.SYSTEM, self.system_prompt)

  def _history_element_text(self, element: Any) -> str:
    return element["content"].strip()

  def _history_element_role(self, element: Any) -> str:
    return element["role"].strip()

class LLamaAPI(LLM_API):
  def __init__(self, system_prompt: str, model_path: str, instance_name: str = "LLaMA", stop_sequences: List[str] = None, 
    temperature: float = 0.75,):
    #super().__init__(name, system_prompt, api_token_env_var="", model_name="llama-local",   instance_name=instance_name)
    #self.llm = Llama(model_path=model_path, n_ctx=2048)  
    #self.add_text_to_history(LLM_API.Role.SYSTEM, system_prompt)
    super().__init__("LLama", system_prompt, api_token_env_var="", model_name=model_path, instance_name=instance_name)
    self.model = model_path
    self.stop_sequences = stop_sequences or []
    self.temperature = temperature
    self.chat_history = [ 
         {"role": "system", "content": system_prompt},
         {"role": "user", "content": "Task: Alice is at (1, 0) and Bob is at (1, 7). Alice's goal is (1, 7) and Bob's goal is (1, 0)."},
    ]
    valid_map = [
    "(1,7)", "(1,6)", "(1,5)", "(1,4)", "(2,4)", 
    "(0,3)", "(1,3)", "(1,2)", "(1,1)", "(1,0)"
    ]
    system_prompt = f"""You are a robot path negotiation assistant.
                        You will be given initial positions, goals, and a list of valid map cells.
                        Only generate one move per agent per step.
                        Each move must:
                           - Be one step in up/down/left/right
                           - Stay inside valid cells only
                           - Avoid collisions

                        Format your response strictly as:
                        Move: Alice moves to (x1, y1), Bob moves to (x2, y2)
                        Valid map cells: {', '.join(valid_map)}
                        Do not include explanation, greetings, or commentary.
                        Wait for the next user prompt before generating the next move.
                         """
    self.llm = Llama(
            system_prompt=system_prompt,
            model_path="/home/bohan/llama_models/tinyllama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",       
            n_ctx=2048,
            n_threads=4,
            n_gpu_layers=20,           
            verbose=False
        )
    self.add_text_to_history(LLM_API.Role.SYSTEM, system_prompt)
  def query(self, prompt: str) -> Tuple[str, Any]:
    self.add_text_to_history(LLM_API.Role.USER, prompt)

    full_prompt = self._build_prompt()
    print("LLM Prompt:")
    print(full_prompt)
    response = self.llm(full_prompt, stop=["@AGREE", "<|user|>", "</s>"], echo=False,  max_tokens=256)
    print("LLM Raw Response:", response)
    output = response["choices"][0]["text"].strip()
    print("Final Output:", output)
    self.add_text_to_history(LLM_API.Role.ASSISTANT, output)
    print("======== Prompt ========")
    print(full_prompt)
    print("======== LLM Response ========")
    print(output)
    return output, []

  def add_text_to_history(self, role: LLM_API.Role, text: str) -> None:
    self.chat_history.append({
      "role": self.role_to_str(role),
      "content": text
    })

  def role_to_str(self, role: LLM_API.Role) -> str:
    return {
      LLM_API.Role.SYSTEM: "system",
      LLM_API.Role.ASSISTANT: "assistant",
      LLM_API.Role.USER: "user"
    }[role]
    
  def _history_element_role(self, role: str) -> str:
    return f"<|{role}|>"

  def _history_element_text(self, message) -> str:
    if isinstance(message, dict) and "content" in message:
        return message["content"].strip()
    elif isinstance(message, str):
        return message.strip()
    else:
        raise ValueError(f"Invalid message format: {message}")

  def _build_prompt(self, max_tokens: int = 2048) -> str:
    prompt_parts = [
    f"{self._history_element_role(msg['role'])}: {self._history_element_text(msg)}"
    for msg in self.chat_history
    ]
    final_prompt = ""
    total_tokens = 0
    for part in reversed(prompt_parts):
        token_count = len(part.split())  
        if total_tokens + token_count > max_tokens:
            break
        final_prompt = part + "\n" + final_prompt
        total_tokens += token_count
    return final_prompt + "<|assistant|>:"
    
def safe_extract_move(response: str):
    import re
    try:
        match = re.search(r"Move:\s*Alice moves to \((\d+), (\d+)\), Bob moves to \((\d+), (\d+)\)", response)
        if not match:
            return None
        ax, ay, bx, by = map(int, match.groups())
        return (("alice", (ax, ay)), ("bob", (bx, by)))
    except Exception as e:
        print(f"[safe_extract_move] Failed to parse move from LLM output: {e}")
        return None
if __name__ == "__main__":
    from src.llms.llm_primitives import LLamaAPI

    llama = LLamaAPI(
        system_prompt="You are a helpful negotiation assistant. Respond only with concise action suggestions.",
        model_path="/home/bohan/llama_models/tinyllama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    )

    response, _ = llama.query("Task: Alice is at (1, 0) and Bob is at (1, 7). Alice's goal is (1, 7) and Bob's goal is (1, 0). What moves should Alice and Bob take to reach their goals?")
    print("Response:", response)
