**Math Solver with Hand Gestures**

This project combines computer vision and generative AI to create an interactive math solver that responds to hand gestures. Users can draw math problems on a digital canvas using hand movements, and the system will attempt to solve them using Google's Gemini AI

**Features**
👆 Draw with your finger: Use your index finger to write math problems on the virtual canvas

👍 Clear the canvas: Show a thumbs-up gesture to erase everything

✋ Solve the problem: Open your hand to send the drawn problem to Gemini AI for solving

📝 Display solutions: The AI's response appears on the canvas in real-time

**Technologies Used**
OpenCV for computer vision and image processing

CVZone for hand tracking and gesture recognition

Google's Gemini AI for mathematical problem solving

NumPy for canvas manipulation

**Gesture Controls**
Gesture	Action
👆 Index finger up	Draw on the canvas
👍 Thumb up	Clear the canvas
✋ Open hand	Solve the drawn math problem

**Requirements**
Python 3.7+
Webcam
Internet connection (for Gemini API access)

**Notes**
The application uses your webcam - ensure you have proper lighting for best results

Draw numbers and symbols clearly for better recognition

The system works best with simple arithmetic problems
