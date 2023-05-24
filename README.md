
# WhatsApp Chat Analysis

This Python code allows you to analyze WhatsApp chat data exported from the WhatsApp application. It performs basic statistical analysis, visualizes message counts over time, and conducts sentiment analysis on text-based messages using the VADER (Valence Aware Dictionary and sEntiment Reasoner) lexicon.

## Prerequisites

- Python 3.x
- pandas
- matplotlib
- nltk

Install the required packages using pip:


## Usage

1. Export your WhatsApp chat from the WhatsApp application in plain text format.

2. Save the exported chat file in the same directory as the Python code and name it `whatsapp_chat.txt`.

3. Run the Python code using the following command:

import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer

# Step 1: Preprocess the data
def preprocess_chat_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    chat_data = []
    for line in lines:
        line = line.strip()
        if line.count("-") >= 2:
            timestamp_end = line.index("-") - 1
            timestamp = line[:timestamp_end]
            content = line[timestamp_end + 3:]

            if ": " in content and content != "<Media omitted>":
                sender, message = content.split(": ", 1)
                chat_data.append((timestamp, sender, message))

    return chat_data

# Step 2: Load and structure the data
chat_file = "whatsapp_chat.txt"  # Replace with the path to your exported chat file
chat_data = preprocess_chat_data(chat_file)
df = pd.DataFrame(chat_data, columns=["Timestamp", "Sender", "Message"])

# Step 3: Explore basic statistics
total_messages = len(df)
participants = df["Sender"].unique()
message_counts = df["Sender"].value_counts()
average_message_length = df["Message"].apply(len).mean()

print("Basic Statistics:")
print("Total messages:", total_messages)
print("Participants:", participants)
print("Message counts:\n", message_counts)
print("Average message length:", average_message_length)

# Step 4: Sentiment analysis
analyzer = SentimentIntensityAnalyzer()
df["Sentiment"] = df["Message"].apply(lambda x: analyzer.polarity_scores(x)["compound"])

# Step 5: Visualize sentiment
plt.figure(figsize=(12, 6))
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df["Date"] = df["Timestamp"].dt.date
sentiment_by_date = df.groupby("Date")["Sentiment"].mean()
sentiment_by_date.plot(kind="line", marker="o")
plt.xlabel("Date")
plt.ylabel("Average Sentiment")
plt.title("Average Sentiment Over Time")
plt.show()

4. The code will display basic statistics about the chat, generate a line plot showing message counts over time, and display a line plot of the average sentiment over time.

## Customization

- If your exported chat file has a different name or location, modify the `chat_file` variable in the code with the appropriate file path.

- You can further customize the code to meet your specific analysis requirements. For example, you can add additional analysis techniques, such as topic modeling or network analysis, to gain deeper insights into your WhatsApp chat data.

## Acknowledgments

- This code uses the `SentimentIntensityAnalyzer` class from the `nltk.sentiment` module for sentiment analysis. Make sure to download the VADER lexicon using the NLTK Downloader before running the code.

- The code also utilizes the `pandas` and `matplotlib` libraries for data manipulation and visualization.
