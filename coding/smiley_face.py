# filename: smiley_face.py
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Create a figure and axes
fig, ax = plt.subplots()

# Face
face = patches.Circle((0, 0), radius=1, facecolor='yellow', edgecolor='black')
ax.add_patch(face)

# Eyes
eye_radius = 0.1
left_eye = patches.Circle((-0.4, 0.4), radius=eye_radius, facecolor='black')
right_eye = patches.Circle((0.4, 0.4), radius=eye_radius, facecolor='black')
ax.add_patch(left_eye)
ax.add_patch(right_eye)

# Mouth
mouth = patches.Arc((0, -0.2), width=0.8, height=0.7, angle=0, theta1=0, theta2=180, color='black') #Corrected theta2
ax.add_patch(mouth)

# Set axis limits
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)

# Set aspect ratio to equal
ax.set_aspect('equal')

# Hide axes
ax.axis('off')

# Show the plot
plt.show()