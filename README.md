Alphabet Learning Projector Game for Kids 🎯

An interactive projector-based game developed using OpenCV and Python, designed to help school children learn alphabets in a fun and engaging way. Players throw a ball at projected letters, and when a hit is detected, the corresponding letter "pops" with a sound, reinforcing learning through both visual and audio feedback.

🚀 Features
Interactive Learning: Children can learn the alphabet by interacting with projected letters on a wall or screen.
Real-Time Hit Detection: Using a camera and OpenCV, the game detects when a ball hits a projected letter.
Audio Feedback: Each time a letter is hit, a corresponding sound is played to reinforce recognition.
Dynamic Slide Transition: New sets of alphabets are displayed after all current letters have been hit.
Calibration Module: Custom calibration to align the camera with the projected area, ensuring accurate hit detection.

🛠️ Technologies Used
OpenCV: For real-time image processing and hit detection.
Python: Backend logic and game control.
Pygame: For playing audio files.

📂 Project Structure
/project-directory
│
├── main.py                # Main script for running the game
├── calibration.py         # Calibration module for setting up the camera and projector
├── display_letters.py     # Module for displaying letters on the slide
├── audio/                 # Directory containing alphabet sound files
├── single_alphabets/      # Directory containing alphabet images
├── bg.png                 # Background image for the slide
├── calibration_data.npz   # Stores calibration data for projector alignment
└── README.md              # Project documentation
