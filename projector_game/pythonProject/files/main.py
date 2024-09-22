import cv2
import numpy as np
import pygame
import random
import os
import time
import pyttsx3

# Initialize Pygame mixer for sound playback
pygame.mixer.init()

# Initialize pyttsx3 for text-to-speech
engine = pyttsx3.init()

# Directory paths
# audio_dir = 'audio/'
image_dir = 'single alphabets/'
special_image_path = 'single alphabets/jewel-5-removebg-preview.png'
#calibration_file = 'calibration_data.npz'

# Verify and load sounds for each letter
'''sounds = {}
for i in range(ord('A'), ord('Z') + 1):
    letter = chr(i)
   sound_path = os.path.join(audio_dir, f'{letter}.wav')
    if os.path.exists(sound_path):
        sounds[letter] = pygame.mixer.Sound(sound_path)
    else:
        print(f"Warning: Sound file {sound_path} not found!")'''

# Verify and load letter images
letter_images = {}
target_size = (100, 100)  # Target size for letter images

# Check if the extracted directory has the correct files
if not os.path.exists(image_dir):
    print(f"Error: Directory {image_dir} not found!")
else:
    for i in range(ord('A'), ord('Z') + 1):
        letter = chr(i)
        img_path = os.path.join(image_dir, f'{letter}.png')

        if os.path.exists(img_path):
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            letter_images[letter] = cv2.resize(img, target_size)  # Resize to target size
        else:
            print(f"Error: Image file {img_path} not found!")

# Load the special image to be displayed
special_image = cv2.imread(special_image_path, cv2.IMREAD_UNCHANGED)
if special_image is None:
    print(f"Error: Special image file {special_image_path} not found!")
    exit()
else:
    special_image = cv2.resize(special_image, target_size)

# Load the calibration data
'''if os.path.exists(calibration_file):
    calibration_data = np.load(calibration_file)
    M = calibration_data['M']
else:
    print(f"Error: Calibration file {calibration_file} not found!")
    exit()'''

# Load the background image
background_image_path = 'bg.png'
background_image = cv2.imread(background_image_path)

if background_image is None:
    print(f"Error: Background image file {background_image_path} not found!")
    exit()

# Resize the background image to the screen size (800x400 for this example)
screen_width, screen_height = 800, 400
background_image = cv2.resize(background_image, (screen_width, screen_height))

# Initialize score
score = 0


# Function to play audio using pyttsx3
def play_audio(letter):
    engine.say(letter)
    engine.runAndWait()


# Function to display a set of 4 letters with increased gaps and circular boundary lines
def display_letters():
    # Use the background image as the slide
    slide = background_image.copy()

    letters = random.sample(list(letter_images.keys()), 4)  # Convert keys to list

    # Calculate the starting x position to center the letters
    gap = 50  # Gap between letters
    total_width = len(letters) * target_size[0] + (len(letters) - 1) * gap
    start_x = (screen_width - total_width) // 2
    y = (screen_height - target_size[1]) // 2  # Center vertically

    circles = []

    for i, letter in enumerate(letters):
        x = start_x + i * (target_size[0] + gap)
        center_x = x + target_size[0] // 2
        center_y = y + target_size[1] // 2
        radius = target_size[0] // 2  # Assuming a square image, so half the width

        # Blend the letter image with the slide using alpha channel
        img = letter_images[letter]
        if img.shape[2] == 4:  # If the image has an alpha channel
            alpha_channel = img[:, :, 3] / 255.0
            for c in range(3):
                slide[y:y + target_size[1], x:x + target_size[0], c] = (alpha_channel * img[:, :, c] +
                                                                        (1 - alpha_channel) * slide[y:y + target_size[1], x:x + target_size[0], c])
        else:
            slide[y:y + target_size[1], x:x + target_size[0]] = img  # Assuming letter images are 100x100 pixels

        circles.append((center_x, center_y, radius))  # Store center and radius of the circle

    return letters, slide, circles


# Function to detect hit using circular boundaries
def detect_hit(frame, circles):
    lower_colors = [
        np.array([35, 100, 100]),  # Green
        np.array([0, 0, 255]),  # White
        #np.array([10, 100, 100]),  # Orange
    ]
    upper_colors = [
        np.array([85, 255, 255]),  # Green
        np.array([180, 255, 255]),  # White
        #np.array([25, 255, 255]),  # Orange

    ]

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = sum([cv2.inRange(hsv, lower, upper) for lower, upper in zip(lower_colors, upper_colors)])

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cx, cy = x + w // 2, y + h // 2  # Center of the contour

        for i, (center_x, center_y, radius) in enumerate(circles):
            if np.sqrt((center_x - cx) ** 2 + (center_y - cy) ** 2) < radius:
                print(f"Hit detected at: ({cx}, {cy}) within circle: center=({center_x}, {center_y}), radius={radius}")  # Debug print
                return i
    return None


# Function to update and display the score
def update_score(slide, score):
    slide_copy = slide.copy()  # Make a copy of the slide to update the score
    cv2.putText(slide_copy, f'SCORE: {score}', (screen_width - 170, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    return slide_copy

# Set up your camera
cap = cv2.VideoCapture(0)

# Display initial set of letters and background
current_letters, slide, circles = display_letters()
hit_states = {letter: False for letter in current_letters}

# Setup window with minimize and close buttons
cv2.namedWindow('Slide', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Slide', cv2.WINDOW_NORMAL, cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Process the frame to detect hits
    hit_index = detect_hit(frame, circles)

    if hit_index is not None and not hit_states[current_letters[hit_index]]:
        hit_letter = current_letters[hit_index]

        # Play the sound for the hit letter using pyttsx3
        play_audio(hit_letter)

        # Replace the hit letter with the special image
        circle_x, circle_y, radius = circles[hit_index]
        x1, y1 = circle_x - radius, circle_y - radius
        x2, y2 = circle_x + radius, circle_y + radius

        # Ensure the coordinates are within bounds
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, screen_width), min(y2, screen_height)

        if special_image.shape[2] == 4:  # If the special image has an alpha channel
            alpha_channel = special_image[:, :, 3] / 255.0
            for c in range(3):
                slide[y1:y2, x1:x2, c] = (alpha_channel * special_image[:, :, c] +
                                          (1 - alpha_channel) * slide[y1:y2, x1:x2, c])
        else:
            slide[y1:y2, x1:x2] = special_image

        hit_states[hit_letter] = True

        # Increment score
        score += 1

        # Small delay after each hit
        time.sleep(0.2)  # Adjust the delay as needed

        # Check if all letters are hit
        if all(hit_states.values()):
            # Delay after all letters are hit before showing the new set
            time.sleep(1)  # Adjust the delay as needed

            current_letters, slide, circles = display_letters()
            hit_states = {letter: False for letter in current_letters}

    # Update and display the score
    slide_with_score = update_score(slide, score)

    # Show the slide
    cv2.imshow('Slide', slide_with_score)

    # Show the frame
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
