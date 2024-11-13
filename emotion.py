# Import SentimentIntensityAnalyzer from nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download the VADER lexicon if not already downloaded
nltk.download('vader_lexicon')

def detect_emotion(text):
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(text)
    
    # Keywords dictionary for specific emotions
    keywords = {
        "happy": "Happy 😀",
        "joy": "Joy 😊",
        "sad": "Sad 😢",
        "angry": "Angry 😠",
        "fear": "Fear 😨",
        "afraid": "Fear 😨",
        "disgust": "Disgust 🤢",
        "shame": "Shame 😞",
        "excited": "Excited 😆",
        "surprised": "Surprised 😲",  # Specific handling for "surprised"
        "bored": "Bored 😒",
        "amazing": "Amazing 🤩",
        "unbelievable": "Surprised 😲",
        "lottery": "Surprised 😲",  # Handle "lottery" related expressions as surprised
    }

    # First, check for specific keywords like 'surprised', 'lottery', etc.
    for keyword, emotion_keyword in keywords.items():
        if keyword.lower() in text.lower():
            emotion = emotion_keyword
            break  # Exit the loop once a keyword is matched
    else:
        # If no keywords match, fall back to sentiment analysis
        if sentiment_scores['compound'] >= 0.05:
            emotion = "Positive 😊"
        elif sentiment_scores['compound'] <= -0.05:
            emotion = "Negative 😢"
        else:
            emotion = "Neutral 😐"

    return {
        "emotion": emotion,
        "scores": sentiment_scores
    }

if __name__ == "__main__":
    print("Enter the paragraph of text to analyze (press Enter twice to finish):")
    lines = []
    while True:
        line = input()
        if line:
            lines.append(line)
        else:
            break
    text = "\n".join(lines)  # Combine lines into a single paragraph

    print("\nText to be analyzed:")
    print(text)
    result = detect_emotion(text)
    print("\nEmotion Detected:", result['emotion'])
    print("Sentiment Scores:", result['scores'])
