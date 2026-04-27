"""Utility classes."""

from typing import TypeVar

import google.genai as genai
import google.genai.types as gentypes
import tenacity
from dotenv import load_dotenv
import os

load_dotenv()

T = TypeVar('T')

# Vertex AI model ids
# https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference
DEFAULT_T2I_VERTEX_ID = 'imagen-3.0-generate-001'
DEFAULT_LLM_VERTEX_ID = 'gemini-2.0-flash'

# Initialize new google-genai client using project ID from environment variable
project_id = os.getenv('VERTEX_PROJECT_ID', 'project-aisdm-489304')

_client = genai.Client(
    vertexai=True,
    project=project_id,
    location='us-central1'
)


class LLM:
  """LLM for text manipulation."""

  def __init__(
      self,
      model_id=DEFAULT_LLM_VERTEX_ID,
      use_json_constraints=False,
  ):
    self.model_id = model_id
    self.use_json_constraints = use_json_constraints

  def generate(self, prompt) -> str:
    """Generates the response from the LLM."""
    config = None
    if self.use_json_constraints:
      config = gentypes.GenerateContentConfig(
          response_mime_type='application/json'
      )
    response = _client.models.generate_content(
        model=self.model_id,
        contents=prompt,
        config=config
    )
    return response.text


class ImageGenerator:
  """Generates images from a prompt."""

  def __init__(self, model_url=DEFAULT_T2I_VERTEX_ID):
    self.model_url = model_url

  @tenacity.retry(stop=tenacity.stop_after_attempt(3))
  def generate_image(self, conversation: str, seed: int):
    """Generate a single image from a prompt."""
    response = _client.models.generate_images(
        model=self.model_url,
        prompt=conversation,
        config=gentypes.GenerateImagesConfig(
            number_of_images=1,
            language='en',
            negative_prompt=(
                'multiple dishes, blurry, painting, cartoon, artificial, nsfw,'
                ' bad quality, bad anatomy, worst quality, low quality, low'
                ' resolutions, extra fingers, blur, blurry, ugly, wrongs'
                ' proportions, watermark, image artifacts, lowres, ugly, jpeg'
                ' artifacts, deformed, noisy image'
            ),
            seed=seed,
            aspect_ratio='1:1',
            safety_filter_level='BLOCK_FEW',
            person_generation='ALLOW_ADULT',
        )
    )
    return response.generated_images[0]

  def generate_diverse_images(self, conversation: str, seed: int):
    """Samples 4 responses with different seeds."""
    response = _client.models.generate_images(
        model=self.model_url,
        prompt=conversation,
        config=gentypes.GenerateImagesConfig(
            number_of_images=4,
            language='en',
            negative_prompt=(
                'multiple dishes, blurry, painting, cartoon, artificial, nsfw,'
                ' bad quality, bad anatomy, worst quality, low quality, low'
                ' resolutions, extra fingers, blur, blurry, ugly, wrongs'
                ' proportions, watermark, image artifacts, lowres, ugly, jpeg'
                ' artifacts, deformed, noisy image'
            ),
            seed=seed,
            aspect_ratio='1:1',
            safety_filter_level='BLOCK_FEW',
            person_generation='ALLOW_ADULT',
        )
    )
    return response.generated_images