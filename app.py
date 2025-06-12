from flask import Flask, abort, render_template, request, jsonify, session, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from sqlalchemy import JSON, Integer
from werkzeug.security import generate_password_hash, check_password_hash
import requests
import os
import random
import re
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///primalfit.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Database Model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

class Progress(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    date = db.Column(db.Date, default=datetime.utcnow)
    weight = db.Column(db.Float)
    body_fat = db.Column(db.Float)
    calories_consumed = db.Column(db.Integer)
    calories_burned = db.Column(db.Integer)
    workout_duration = db.Column(db.Integer)
    sleep_hours = db.Column(db.Float)  
    sleep_quality = db.Column(db.Integer)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# API Keys
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
STABILITY_API_KEY = os.getenv('STABILITY_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PEXELS_API_KEY = os.getenv('PEXELS_API_KEY')
UNSPLASH_ACCESS_KEY = os.getenv('UNSPLASH_ACCESS_KEY')

BASE_PROMPT = [{
    "role": "system",
    "content": """
    You are Primal, an AI fitness assistant for Primal Fit. 
    Respond in a friendly, motivational tone with concise answers.
    Format responses using clear bullet points and simple headings.
    Never use markdown or special formatting.
    Focus on fitness-related topics only.
    Ask for user preferences to create personalized plans.
    """
}]

# Security Functions
def clean_response(text):
    text = re.sub(r'\*\*|\*|`', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    return text.strip()

# Image Generation Functions
def generate_ai_image(query):
    try:
        # Create necessary directories
        os.makedirs("static/generated", exist_ok=True)
        os.makedirs("static/images/fallback", exist_ok=True)

        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"generated/{timestamp}_{query[:20]}.png"
        full_path = os.path.join("static", filename)

        # Try external services
        services = [
            _try_stability_ai,
            _try_dalle,
            _try_pexels,
            _try_unsplash
        ]
        random.shuffle(services)

        for service in services:
            try:
                image_url = service(query)
                if image_url:
                    if image_url.startswith("http"):
                        # Download and save external images
                        response = requests.get(image_url, timeout=10)
                        response.raise_for_status()
                        with open(full_path, "wb") as f:
                            f.write(response.content)
                    else:
                        # Use local images directly
                        return url_for('static', filename=image_url)
                    
                    return url_for('static', filename=filename)
            except Exception as e:
                print(f"Error with {service.__name__}: {str(e)}")

        # Ultimate fallback
        return url_for('static', filename='images/fallback/general.jpg')
        
    except Exception as e:
        print(f"Image Generation Error: {str(e)}")
        return url_for('static', filename='images/fallback/general.jpg')

# Update the Stability AI function
def _try_stability_ai(query):
    if not STABILITY_API_KEY: return None
    url = "https://api.stability.ai/v2beta/stable-image/generate/core"
    headers = {"Authorization": f"Bearer {STABILITY_API_KEY}"}
    
    files = {
        "prompt": (None, query),
        "output_format": (None, "png"),
        "model": (None, "sd3"),
        "aspect_ratio": (None, "16:9")
    }

    try:
        response = requests.post(url, headers=headers, files=files, timeout=20)
        response.raise_for_status()
        return response.content  # Return binary content instead of saving here
    except Exception as e:
        print(f"Stability AI Error: {str(e)}")
        return None

def _try_dalle(query):
    if not OPENAI_API_KEY: return None
    url = "https://api.openai.com/v1/images/generations"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "prompt": query,
        "n": 1,
        "size": "1024x1024",
        "model": "dall-e-3",
        "quality": "standard"
    }
    try:
        response = requests.post(url, headers=headers, json=data, timeout=10)
        response.raise_for_status()
        return response.json()['data'][0]['url']
    except Exception as e:
        print(f"DALL-E Error: {str(e)}")
        return None

def _try_pexels(query):
    if not PEXELS_API_KEY: return None
    url = f"https://api.pexels.com/v1/search?query={query}&per_page=1"
    headers = {"Authorization": PEXELS_API_KEY}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()['photos'][0]['src']['large']

def _try_unsplash(query):
    if not UNSPLASH_ACCESS_KEY: return None
    url = f"https://api.unsplash.com/photos/random"
    params = {
        "query": query,
        "client_id": UNSPLASH_ACCESS_KEY,
        "orientation": "landscape"
    }
    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        return response.json()['urls']['regular']
    except Exception as e:
        print(f"Unsplash Error: {str(e)}")
        return None

def get_ai_adaptation(prompt):
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    data = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 500
    }
    
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json=data
    )
    return response.json()['choices'][0]['message']['content']

def parse_ai_response(text):
    # Add error handling and default structure
    try:
        exercises = []
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        current_exercise = {}
        for line in lines:
            if line.lower().startswith('exercise:'):
                if current_exercise:
                    exercises.append(current_exercise)
                current_exercise = {
                    'name': line.split(': ')[1],
                    'type': 'General',
                    'sets': '3',
                    'reps': '10-12',
                    'intensity': 60
                }
            elif line.lower().startswith('type:'):
                current_exercise['type'] = line.split(': ')[1]
            elif line.lower().startswith('sets:'):
                current_exercise['sets'] = line.split(': ')[1]
            elif line.lower().startswith('reps/duration:'):
                current_exercise['reps'] = line.split(': ')[1]
            elif line.lower().startswith('intensity:'):
                current_exercise['intensity'] = int(line.split(': ')[1].replace('%', ''))

        if current_exercise:
            exercises.append(current_exercise)

        return {
            'workout': {'exercises': exercises},
            'adaptations': {'feedback': 'Great start! Focus on maintaining proper form.'}
        }
    except Exception as e:
        # Return default workout structure if parsing fails
        return {
            'workout': {
                'exercises': [{
                    'name': 'Bodyweight Squats',
                    'type': 'Strength',
                    'sets': '3',
                    'reps': '12-15',
                    'intensity': 60,
                    'image': generate_ai_image("bodyweight squats proper form")
                }]
            },
            'adaptations': {'feedback': 'Default workout generated - focus on perfecting form!'}
        }

# Context processor to make function available in templates
@app.context_processor
def inject_ai_functions():
    return dict(generate_ai_image=generate_ai_image)

# Routes
@app.route('/')
@login_required
def home():
    return render_template('index.html',
                         hero_image=generate_ai_image("fitness motivation"),
                         about_image=generate_ai_image("gym equipment"),
                         music_image=generate_ai_image("workout music"),
                         podcast_image=generate_ai_image("fitness podcast"))

@app.route('/chatbot')
def chatbot():
    if "messages" not in session:
        session["messages"] = BASE_PROMPT.copy()
    return render_template('chatbot.html',
                         chatbot_image=generate_ai_image("fitness chatbot"))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    
    if request.method == 'POST':
        name = request.form['name']
        age = int(request.form['age'])
        email = request.form['email']
        password = generate_password_hash(request.form['password'])
        
        if User.query.filter_by(email=email).first():
            flash('Email already exists!')
            return redirect(url_for('register'))
        
        new_user = User(name=name, age=age, email=email, password=password)
        db.session.add(new_user)
        db.session.commit()
        
        flash('Registration successful! Please login.')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('home'))
        
        flash('Invalid credentials!')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/terms')
def terms():
    return render_template('terms.html')

@app.route('/privacy')
def privacy():
    return render_template('privacy.html')

# API Endpoints
@app.route('/api/chat', methods=['POST'])
def chat():
    if "messages" not in session:
        session["messages"] = BASE_PROMPT.copy()
    
    user_message = request.json.get('message', '')
    session["messages"].append({"role": "user", "content": user_message})
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": session["messages"],
        "temperature": 0.7,
        "max_tokens": 500
    }
    
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        ai_response = clean_response(response.json()['choices'][0]['message']['content'])
    except Exception as e:
        ai_response = f"Sorry, I encountered an error: {str(e)}"
    
    session["messages"].append({"role": "assistant", "content": ai_response})
    session.modified = True
    
    return jsonify({
        "response": f'<span style="color: white;">{ai_response}</span>',
        "image": generate_ai_image(user_message) if "workout" in user_message.lower() else None
    })

@app.route('/api/voice', methods=['POST'])
def handle_voice():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file"}), 400
    
    try:
        audio_file = request.files['audio']
        filename = f"static/temp/audio_{datetime.now().timestamp()}.webm"
        audio_file.save(filename)
        
        # Implement speech-to-text processing here
        text = "Voice processing placeholder"
        
        os.remove(filename)
        return jsonify({"text": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/clear_chat', methods=['POST'])
def clear_chat():
    session["messages"] = BASE_PROMPT.copy()
    session.modified = True
    return jsonify({"status": "success"})

@app.route('/api/get-image')
def get_image():
    try:
        query = request.args.get('query', 'fitness')
        image_url = generate_ai_image(query)
        return jsonify({'url': image_url})
    except Exception as e:
        print(f"Image API Error: {str(e)}")
        return jsonify({'url': url_for('static', filename='images/fallback/general.jpg')})

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        # Process form data here
        name = request.form.get('name')
        email = request.form.get('email')
        subject = request.form.get('subject')
        message = request.form.get('message')
        
        # Add your email sending logic here
        flash('Message sent successfully!', 'success')
        return redirect(url_for('contact'))
    
    return render_template('contact.html')

@app.route('/nutrition')
@login_required
def nutrition():
    return render_template('nutrition.html')

@app.route('/api/generate-nutrition-plan', methods=['POST'])
@login_required
def generate_nutrition_plan():
    try:
        data = request.json
        required_fields = ['weight', 'height', 'calories', 'diet_type']
        if not all(field in data for field in required_fields):
            return jsonify({"error": "Missing required fields"}), 400

        prompt = f"""Create a detailed nutrition plan with these specifications:
        - Weight: {data['weight']} kg
        - Height: {data['height']} cm
        - Target Calories: {data['calories']}
        - Diet Type: {data['diet_type'].capitalize()}
        - Restrictions: {data.get('allergies', 'none')}
        - Special Instructions: {data.get('custom_prompt', 'none')}

        Format with these exact section headers:
        [BMI Analysis] (Just the calculated BMI number and classification and some information/motivation in a sentence)
        [Macronutrients] (Protein/Carbs/Fats percentages only)
        [Daily Meal Plan] (Breakfast/Lunch/Dinner/Snacks)
        [Weekly Diet Plan] (7-day meal overview)
        [Grocery List] (Bulleted list)
        [Prep Tips] (Numbered steps, also include ways to better diet, for eg: water intake, workouts etc.)

        Exclude formulas and explanations. Use clear bullet points."""

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }

        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 1500
            }
        )
        response.raise_for_status()

        raw_content = response.json()['choices'][0]['message']['content']
        cleaned_plan = clean_response(raw_content)
        bmi = calculate_bmi(float(data['weight']), float(data['height']))

        return jsonify({
            "plan": cleaned_plan,
            "bmi": bmi,
            "bmi_note": get_bmi_note(bmi) if bmi else ""
        })

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"AI API Error: {str(e)}"}), 502
    except Exception as e:
        return jsonify({"error": f"Server Error: {str(e)}"}), 500

def get_bmi_note(bmi):
    if bmi < 18.5:
        return "Underweight - Consider increasing calorie intake"
    elif 18.5 <= bmi < 25:
        return "Healthy weight - Maintain your balance"
    elif 25 <= bmi < 30:
        return "Overweight - Consider gradual weight loss"
    else:
        return "Obese - Consult a healthcare professional"

def calculate_bmi(weight, height):
    try:
        height_m = height / 100
        return round(weight / (height_m ** 2), 1)
    except ZeroDivisionError:
        return None

@app.route('/workouts')
@login_required
def workouts():
    return render_template('workouts.html',
                         workout_image=generate_ai_image("personalized workout"),
                         progress_image=generate_ai_image("fitness progress tracking"))

@app.route('/api/generate-workout-plan', methods=['POST'])
@login_required
def generate_workout_plan():
    try:
        data = request.json
        required_fields = ['fitness_level', 'workout_type', 'available_equipment', 'weekly_sessions']
        if not all(field in data for field in required_fields):
            return jsonify({"error": "Missing required fields"}), 400

        prompt = f"""Create a detailed workout plan with these parameters:
        Fitness Level: {data['fitness_level']}
        Primary Goal: {data['workout_type']}
        Available Equipment: {data['available_equipment']}
        Sessions per Week: {data['weekly_sessions']}

        Provide the response in EXACTLY this format:
        
        [Workout Schedule]
        Day 1: [Exercise1], [Exercise2], [Exercise3]
        Day 2: [Exercise4], [Exercise5], [Exercise6]
        
        [Exercise Details]
        • [Exercise1]: [Muscle Group] - [Sets]x[Reps] - [Description]
        • [Exercise2]: [Muscle Group] - [Sets]x[Reps] - [Description]
        
        [Progression Plan]
        - Week 1: [Details]
        - Week 2: [Details]"""

        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 1500
            }
        )
        response.raise_for_status()

        raw_content = response.json()['choices'][0]['message']['content']
        cleaned_plan = clean_response(raw_content)
        
        # Extract exercise names for images
        exercises = list(set(
            re.findall(r'• (.*?):', cleaned_plan) +
            re.findall(r'Day \d+: (.*)', cleaned_plan)[0].split(', ')
        ))
        exercise_images = {
            ex.lower().replace(' ', '_'): generate_ai_image(f"{ex} exercise proper form")
            for ex in exercises if ex
        }

        return jsonify({
            "plan": cleaned_plan,
            "exercise_images": exercise_images,
            "status": "success"
        })

    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 500
    
@app.route('/aboutus')
def aboutus():
    return render_template('aboutus.html')  

@app.route('/progress')
@login_required
def progress():
    goals = {
        'body_fat': 15,
        'calories': 2000,
        'sleep_hours': 8  # Add sleep goal
    }
    
    progress_data = Progress.query.filter_by(user_id=current_user.id)\
        .order_by(Progress.date.asc()).limit(30).all()
    
    # Prepare chart data
    dates = [entry.date.strftime('%Y-%m-%d') for entry in progress_data]
    weights = [entry.weight for entry in progress_data]
    calories_consumed = [entry.calories_consumed for entry in progress_data]
    calories_burned = [entry.calories_burned for entry in progress_data]
    sleep_hours = [entry.sleep_hours for entry in progress_data if entry.sleep_hours]
    sleep_quality = [entry.sleep_quality for entry in progress_data if entry.sleep_quality]
    
    return render_template('progress.html',
                         goals=goals,
                         progress_data=progress_data,
                         dates=dates,
                         weights=weights,
                         calories_consumed=calories_consumed,
                         calories_burned=calories_burned,
                         sleep_hours=sleep_hours,
                         sleep_quality=sleep_quality)

@app.route('/api/submit-progress', methods=['POST'])
@login_required
def submit_progress():
    try:
        data = request.json
        new_entry = Progress(
            user_id=current_user.id,
            weight=float(data['weight']),
            body_fat=float(data['body_fat']),
            calories_consumed=int(data['calories_consumed']),
            calories_burned=int(data['calories_burned']),
            workout_duration=int(data['workout_duration']),
            sleep_hours=float(data.get('sleep_hours', 0)) if data.get('sleep_hours') else None,
            sleep_quality=int(data.get('sleep_quality')) if data.get('sleep_quality') else None
        )
        db.session.add(new_entry)
        db.session.commit()
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    os.makedirs("static/generated", exist_ok=True)
    os.makedirs("static/temp", exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)