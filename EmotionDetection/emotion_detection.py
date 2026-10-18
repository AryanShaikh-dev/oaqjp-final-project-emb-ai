"""Emotion detection using the Watson NLP EmotionPredict service."""
import json
import requests

URL = ("https://sn-watson-emotion.labs.skills.network/v1/"
       "watson.runtime.nlp.v1/NlpService/EmotionPredict")
HEADERS = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}
EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness"]


def emotion_detector(text_to_analyse):
    """Analyse text and return emotion scores plus the dominant emotion.

    Returns a dict with None values when the service responds with status 400
    (e.g. blank input).
    """
    payload = {"raw_document": {"text": text_to_analyse}}
    response = requests.post(URL, json=payload, headers=HEADERS, timeout=10)

    # Task 7: error handling for blank / invalid input
    if response.status_code == 400:
        result = {emotion: None for emotion in EMOTIONS}
        result["dominant_emotion"] = None
        return result

    formatted = json.loads(response.text)
    scores = formatted["emotionPredictions"][0]["emotion"]
    result = {emotion: scores[emotion] for emotion in EMOTIONS}
    result["dominant_emotion"] = max(result, key=result.get)
    return result