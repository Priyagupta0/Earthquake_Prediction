from flask import Flask, request, url_for, redirect, render_template, jsonify, flash
import pickle
import numpy as np
import os
import traceback

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Required for flashing messages

# Load the model with error handling
model = None
try:
    # Get the absolute path to the model file
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.pkl')
    print(f"Looking for model at: {model_path}")
    
    if os.path.exists(model_path):
        print("Model file found. Loading model...")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print("Model loaded successfully!")
        
        # Verify the model has the predict method
        if not hasattr(model, 'predict'):
            raise AttributeError("The loaded model does not have a 'predict' method.")
        print("Model verified with predict method.")
    else:
        print("Error: model.pkl not found at the expected location.")
        print("Current working directory:", os.getcwd())
        print("Files in directory:", os.listdir('.'))
        
except Exception as e:
    print(f"\n=== ERROR LOADING MODEL ===")
    print(f"Type: {type(e).__name__}")
    print(f"Error: {str(e)}")
    print("\nStack trace:")
    import traceback
    traceback.print_exc()
    print("\nPlease ensure model.pkl exists and is a valid scikit-learn model.")
    model = None

@app.route('/')
def home():
    return render_template('homepage.html')

@app.route('/home')
def home2():
    return render_template('homepage.html')

@app.route('/aboutproject')
def aboutproject():
    return render_template('aboutproject.html')

@app.route('/creator')
def creator():
    return render_template('creator.html')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    print("\n=== Prediction Route ===")
    print(f"Method: {request.method}")
    print(f"Headers: {dict(request.headers)}")
    
    if request.method == 'GET':
        print("GET request received, redirecting to home")
        return redirect(url_for('home'))
    
    # Handle POST request
    try:
        # Check if it's an AJAX request
        is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest' or 'application/json' in request.headers.get('Accept', '')
        print(f"Is AJAX: {is_ajax}")
        print(f"Form data: {request.form}")
        
        # Check if model is loaded
        if model is None:
            error_msg = 'Error: Prediction model not loaded. Please try again later.'
            print(error_msg)
            if is_ajax:
                return jsonify({'success': False, 'message': error_msg}), 500
            flash(error_msg, 'error')
            return redirect(url_for('home'))
        
        # Get and validate form data
        try:
            latitude = float(request.form.get('a', 0))
            longitude = float(request.form.get('b', 0))
            depth = float(request.form.get('c', 0))
            
            # Input validation
            if not (-90 <= latitude <= 90):
                raise ValueError("Latitude must be between -90 and 90 degrees")
            if not (-180 <= longitude <= 180):
                raise ValueError("Longitude must be between -180 and 180 degrees")
            if depth < 0:
                raise ValueError("Depth must be a positive number")
                
            # Prepare input for model (same order as training: Latitude, Longitude, Depth)
            input_data = np.array([[latitude, longitude, depth]])
            print(f"Making prediction with input: {input_data}")
            
            # Get prediction (this will be an integer class: 2, 3, 4, or 5)
            predicted_class = int(model.predict(input_data)[0])
            
            # Convert class back to a magnitude value (using the middle of the range for display)
            # Class 2: 2.0-2.9 -> 2.5
            # Class 3: 3.0-3.9 -> 3.5
            # Class 4: 4.0-4.9 -> 4.5
            # Class 5: 5.0-5.9 -> 5.5
            output = predicted_class + 0.5  # This gives us the middle of the range
            
            print(f"Prediction - Input: {input_data}, Predicted class: {predicted_class}, Output magnitude: {output}")
            
        except ValueError as e:
            error_msg = f'Invalid input: {str(e)}'
            print(f"Validation error: {error_msg}")
            if is_ajax:
                return jsonify({'success': False, 'message': error_msg}), 400
            flash(error_msg, 'error')
            return redirect(url_for('home'))
        
        # Determine risk level based on predicted class
        if predicted_class < 4:
            risk = 'No'
            risk_class = 'no-risk'
        elif 4 <= predicted_class < 6:
            risk = 'Low'
            risk_class = 'low-risk'
        elif 6 <= predicted_class < 8:
            risk = 'Moderate'
            risk_class = 'moderate-risk'
        else:
            risk = 'High'
            risk_class = 'high-risk'
        
        # Prepare result data
        result = {
            'magnitude': output,
            'risk': risk,
            'risk_class': risk_class,
            'location': f"Lat: {latitude:.4f}°, Long: {longitude:.4f}°",
            'depth': f"{depth:.1f} km"
        }
        
        print(f"Prediction successful. Risk level: {risk}")
        
        if is_ajax:
            # For AJAX requests, return JSON with redirect URL
            redirect_url = url_for('prediction_result', 
                                 magnitude=result['magnitude'],
                                 risk=result['risk'])
            return jsonify({
                'success': True,
                'redirect': redirect_url,
                'result': result
            })
        else:
            # For regular form submission, render the template directly
            return render_template('prediction.html',
                                p=result['magnitude'],
                                q=result['risk'],
                                result=result)
    
    except Exception as e:
        error_msg = 'An unexpected error occurred. Please try again.'
        print(f"Unexpected error: {str(e)}")
        print(traceback.format_exc())
        
        if is_ajax:
            return jsonify({
                'success': False,
                'message': error_msg
            }), 500
        
        flash(error_msg, 'error')
        return redirect(url_for('home'))
                            
    except Exception as e:
        error_msg = 'An unexpected error occurred. Please try again.'
        print(f"{error_msg} Error: {str(e)}")
        if is_ajax:
            return jsonify({'success': False, 'message': error_msg}), 500
        flash(error_msg, 'error')
        return redirect(url_for('home'))

@app.route('/prediction/result')
def prediction_result():
    """
    Display the prediction result page with the given magnitude and risk level.
    This allows users to directly link to or share prediction results.
    """
    try:
        # Get parameters from the URL
        magnitude = request.args.get('magnitude', '0.0')
        risk = request.args.get('risk', 'Unknown')
        
        # Convert magnitude to float for validation
        try:
            mag_float = float(magnitude)
        except (ValueError, TypeError):
            mag_float = 0.0
            
        # Set default risk class based on the risk level
        risk_lower = str(risk).lower()
        if 'no' in risk_lower:
            risk_class = 'no-risk'
        elif 'low' in risk_lower:
            risk_class = 'low-risk'
        elif 'moderate' in risk_lower:
            risk_class = 'moderate-risk'
        elif 'high' in risk_lower:
            risk_class = 'high-risk'
        else:
            risk_class = 'very-high-risk'
        
        # Create a result dictionary to pass to the template
        result = {
            'magnitude': f"{mag_float:.2f}",
            'risk': risk,  # Changed from 'risk_level' to 'risk' for consistency
            'risk_class': risk_class,
            'location': 'Location not specified',
            'depth': 'Depth not specified'
        }
        
        print(f"Rendering prediction result: {result}")
        
        # Render the template with the result
        return render_template('prediction.html', 
                            p=result['magnitude'],
                            q=result['risk'],  # Changed from 'risk_level' to 'risk'
                            result=result)
                            
    except Exception as e:
        print(f"Error in prediction_result: {str(e)}")
        print(traceback.format_exc())
        flash('Error displaying prediction result. Please try again.', 'error')
        return redirect(url_for('home'))

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)