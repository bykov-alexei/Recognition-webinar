import face_recognition
import matplotlib.pyplot as plt
import matplotlib.patches as patches
image = face_recognition.load_image_file("rgb.png")
face_locations = face_recognition.face_locations(image)
for location in face_locations:
    top, right, bottom, left = location
    x, y = left, top
    w, h = (right - left), (bottom - top)

fig, ax = plt.subplots(1)
ax.axis('off')
rect = patches.Rectangle((x, y), w, h, linewidth=2,edgecolor='r',facecolor='none')
ax.add_patch(rect)
ax.imshow(image)
plt.savefig('detected.jpg')