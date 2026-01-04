"""
Firebase Authentication Module for Parkinson's Detection System
"""
import pyrebase
import streamlit as st
from datetime import datetime

# Firebase Configuration
firebase_config = {
    "apiKey": "AIzaSyBarQqLsLBqZDZN06AskQ0wLB3Kld1Q9yA",
    "authDomain": "parkinson-disease-detect-af923.firebaseapp.com",
    "projectId": "parkinson-disease-detect-af923",
    "storageBucket": "parkinson-disease-detect-af923.firebasestorage.app",
    "messagingSenderId": "1026157711844",
    "appId": "1:1026157711844:web:ce9bee6f4aa036d66cf7f1",
    "databaseURL": "https://parkinson-disease-detect-af923-default-rtdb.firebaseio.com/"
}

# Initialize Firebase
firebase = pyrebase.initialize_app(firebase_config)
auth = firebase.auth()
db = firebase.database()

def init_session_state():
    """Initialize session state variables"""
    if 'user' not in st.session_state:
        st.session_state.user = None
    if 'user_email' not in st.session_state:
        st.session_state.user_email = None
    if 'bmdc_reg_no' not in st.session_state:
        st.session_state.bmdc_reg_no = None

def signup_user(email, password, bmdc_reg_no, full_name):
    """Sign up a new user with BMDC registration number"""
    try:
        # Create user account
        user = auth.create_user_with_email_and_password(email, password)
        
        # Store user info in session state (database storage is optional)
        user_data = {
            'email': email,
            'bmdc_reg_no': bmdc_reg_no,
            'full_name': full_name,
            'created_at': str(datetime.now()),
            'role': 'doctor'
        }
        
        # Try to store in database, but don't fail if database is not available
        try:
            db.child('users').child(user['localId']).set(user_data)
        except Exception as db_error:
            # Database storage failed, but authentication succeeded
            # Store in session state as fallback
            print(f"Warning: Database storage failed: {db_error}")
        
        # Store in session state immediately
        if 'user_profiles' not in st.session_state:
            st.session_state.user_profiles = {}
        st.session_state.user_profiles[user['localId']] = user_data
        
        # Also set the current session variables for immediate login
        st.session_state.user = user
        st.session_state.user_email = email
        st.session_state.bmdc_reg_no = bmdc_reg_no
        st.session_state.full_name = full_name
        
        # Send email verification
        try:
            auth.send_email_verification(user['idToken'])
        except:
            pass  # Email verification is optional
        
        return True, "Account created successfully! Logging you in..."
    except Exception as e:
        error_message = str(e)
        if "EMAIL_EXISTS" in error_message:
            return False, "Email already exists. Please use a different email or login."
        elif "WEAK_PASSWORD" in error_message:
            return False, "Password is too weak. Use at least 6 characters."
        elif "INVALID_EMAIL" in error_message:
            return False, "Invalid email format."
        else:
            return False, f"Signup failed: {error_message}"

def login_user(email, password):
    """Login user"""
    try:
        user = auth.sign_in_with_email_and_password(email, password)
        
        # Initialize user_profiles if not exists
        if 'user_profiles' not in st.session_state:
            st.session_state.user_profiles = {}
        
        # Try to get user data from database, use fallback if not available
        user_data = None
        try:
            user_db_data = db.child('users').child(user['localId']).get()
            if user_db_data.val():
                user_data = user_db_data.val()
        except Exception as db_error:
            print(f"Warning: Database read failed: {db_error}")
        
        # Use stored profile or create default
        if not user_data:
            if user['localId'] in st.session_state.user_profiles:
                user_data = st.session_state.user_profiles[user['localId']]
            else:
                # Default user data if database is not available
                user_data = {
                    'email': email,
                    'bmdc_reg_no': 'N/A',
                    'full_name': email.split('@')[0].title()
                }
        
        st.session_state.user = user
        st.session_state.user_email = email
        st.session_state.bmdc_reg_no = user_data.get('bmdc_reg_no', 'N/A')
        st.session_state.full_name = user_data.get('full_name', 'User')
        return True, "Login successful!"
    except Exception as e:
        error_message = str(e)
        if "INVALID_LOGIN_CREDENTIALS" in error_message or "INVALID_PASSWORD" in error_message:
            return False, "Invalid email or password."
        elif "EMAIL_NOT_FOUND" in error_message:
            return False, "Email not found. Please sign up first."
        elif "USER_DISABLED" in error_message:
            return False, "This account has been disabled."
        else:
            return False, f"Login failed: {error_message}"

def reset_password(email):
    """Send password reset email"""
    try:
        auth.send_password_reset_email(email)
        return True, "Password reset email sent! Please check your inbox."
    except Exception as e:
        error_message = str(e)
        if "EMAIL_NOT_FOUND" in error_message:
            return False, "Email not found."
        elif "INVALID_EMAIL" in error_message:
            return False, "Invalid email format."
        else:
            return False, f"Password reset failed: {error_message}"

def logout_user():
    """Logout user"""
    st.session_state.user = None
    st.session_state.user_email = None
    st.session_state.bmdc_reg_no = None
    st.session_state.full_name = None

def is_logged_in():
    """Check if user is logged in"""
    return st.session_state.user is not None

def get_user_info():
    """Get current user information"""
    return {
        'email': st.session_state.get('user_email', 'N/A'),
        'bmdc_reg_no': st.session_state.get('bmdc_reg_no', 'N/A'),
        'full_name': st.session_state.get('full_name', 'User')
    }
