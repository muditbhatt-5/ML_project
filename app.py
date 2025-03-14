from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("./Data/mood_based_movie_data.csv")

# Map mood to integers
unique_moods = df['Mood'].unique()
mood_to_int = {mood: i for i, mood in enumerate(unique_moods)}
df['Mood'] = df['Mood'].map(mood_to_int)

# Extract features and target variable
X = df[['Mood']].values
y = df['Movie_Name'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM with RBF kernel
svm = SVC(kernel='rbf', gamma=0.1, C=1.0, probability=True)
svm.fit(X_train, y_train)

# Save the trained model
with open("./Model/svm_model.pkl", "wb") as f:
    pickle.dump(svm, f)

# Flask app
app = Flask(__name__, template_folder='./templates')

def generate_plot(plot_type):
    """Generate a base64-encoded plot based on the requested type."""
    plt.figure(figsize=(6, 4))
    plt.clf()  # Clear figure before plotting new one

    if plot_type == 'scatter':
        plt.scatter(X, y, c='blue', alpha=0.5)
        plt.xlabel('Mood')
        plt.ylabel('Movie')
        plt.title('Scatter Plot')
    elif plot_type == 'bar':
        mood_counts = df['Mood'].value_counts()
        mood_counts.plot(kind='bar', color='green')
        plt.xlabel('Mood')
        plt.ylabel('Count')
        plt.title('Bar Plot')
    elif plot_type == 'hist':
        df['Mood'].plot(kind='hist', bins=10, color='purple', alpha=0.7)
        plt.xlabel('Mood')
        plt.ylabel('Frequency')
        plt.title('Histogram')
    elif plot_type == 'line':
        df.groupby('Mood').size().plot(kind='line', marker='o', color='red')
        plt.xlabel('Mood')
        plt.ylabel('Count')
        plt.title('Line Plot')
    elif plot_type == 'box':
        df[['Mood']].boxplot()
        plt.title('Box Plot')
    else:
        return None
    
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return plot_url

@app.route('/')
def home():
    return render_template('index.html', moods=list(mood_to_int.keys()), initial_plot=generate_plot('scatter'))

@app.route('/plot', methods=['POST'])
def plot():
    plot_type = request.form.get('plot_type')
    plot_image = generate_plot(plot_type)
    if plot_image:
        return jsonify({'plot_image': plot_image})
    else:
        return jsonify({'error': 'Invalid plot type'})

@app.route('/predict_movie', methods=['POST'])
def predict_movie():
    mood = request.form.get('mood')
    mood_encoded = mood_to_int.get(mood, -1)
    if mood_encoded == -1:
        return jsonify({'movie': 'Unknown'})
    
    recommended_movie = df[df['Mood'] == mood_encoded]['Movie_Name'].sample(1).values[0]
    
    return jsonify({'movie': recommended_movie})

if __name__ == '__main__':
    app.run(debug=True)
