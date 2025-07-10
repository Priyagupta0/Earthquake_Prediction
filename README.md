# 🌍 Earthquake Prediction System

A machine learning-based web application that predicts earthquake magnitudes and assesses risk levels based on seismic data. Built with Python, Flask, and scikit-learn.

![Earthquake Prediction System](https://img.shields.io/badge/status-active-success)
![Python](https://img.shields.io/badge/python-3.7+-blue.svg)
![Flask](https://img.shields.io/badge/flask-2.0.1-green.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0.2-orange.svg)

## 📋 Features

- **Real-time Prediction**: Get instant earthquake magnitude predictions
- **Risk Assessment**: Categorizes risk levels (No, Low, Moderate, High, Very High)
- **Responsive Design**: Works seamlessly on all devices
- **Safety Recommendations**: Provides tailored safety tips based on risk level
- **Interactive UI**: Clean and intuitive user interface

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone < github repo link>
   cd earthquake-predictor
   ```

2. **Create and activate a virtual environment**
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Open in browser**
   ```
   http://localhost:5000
   ```

## 🛠️ Project Structure

```
earthquake-predictor/
├── app.py                # Flask application
├── detector.py           # ML model training
├── model.pkl             # Trained model
├── requirements.txt      # Python dependencies
├── static/               # Static files (CSS, JS, images)
│   └── style.css        # Main stylesheet
└── templates/            # HTML templates
    ├── homepage.html    # Main page with prediction form
    └── prediction.html # Results page
```

## 🤖 Machine Learning Model

The prediction system uses a Random Forest Classifier trained on historical seismic data. The model takes the following inputs:

- Latitude
- Longitude
- Depth

And predicts the earthquake magnitude, which is then categorized into risk levels.

## 🌟 Features in Detail

### Prediction Form
- Input fields for latitude, longitude, and depth
- Real-time validation
- Responsive design for all screen sizes

### Results Page
- Clear display of predicted magnitude
- Color-coded risk assessment
- Detailed safety recommendations
- Shareable results

## 📱 Responsive Design

- Fully responsive layout that works on all devices
- Mobile-first approach
- Optimized touch targets for mobile users
- Adaptive font sizes and spacing

## 🧪 Testing

To run the tests:

```bash
python -m pytest tests/
```

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Built with Flask and scikit-learn
- Icons by Font Awesome
- Google Fonts for typography

## 📬 Contact

For any questions or feedback, please contact Me at pg7108970@gmail.com 

---

<div align="center">
  Made with ❤️ by Priya Gupta
</div>
