import cv2
import os   
from flask import Flask, render_template, request, redirect, url_for, session
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import csv
import joblib

# Defining Flask App
app = Flask(__name__)
app.secret_key = "a6f9b1a3c9d4e8g2h7j0k3p1q2r4t5"
nimgs = 10

# Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")


# Initialize Haarcascade
face_detector_path = r'C:\Users\lenovo\Desktop\face_recog\face-recognition-based-attendance-system-master\haarcascade_frontalface_default.xml'
face_detector = cv2.CascadeClassifier(face_detector_path)
if not os.path.exists(face_detector_path):
    raise FileNotFoundError(f"{face_detector_path} not found. Please add it to the project directory.")
face_detector = cv2.CascadeClassifier(face_detector_path)


# If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time')


# get a number of total registered users
def totalreg(user):
    return len(os.listdir(f'static/{user}'))


# extract the face from an image
def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        return face_points
    except:
        return []


# Identify face using ML model
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)


# A function which trains the model on all the faces available in faces folder
def train_model(username):
    faces = []
    labels = []
    userlist = os.listdir(f'static/{username}')
    for user in userlist:
        for imgname in os.listdir(f'static/{username}/{user}'):
            img = cv2.imread(f'static/{username}/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')


# Extract info from today's attendance file in attendance folder
def extract_attendance(user):
    attendance_file = f'Attendance/{user}/Attendance-{datetoday}.csv'

    # Ensure the attendance file exists with proper headers
    if not os.path.exists(attendance_file):
        # Create the file if it doesn't exist, and add the correct headers
        with open(attendance_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Name', 'Roll', 'Time'])  # Ensure correct headers

    # Read the attendance file
    df = pd.read_csv(attendance_file)

    # Check if required columns are present
    required_columns = ['Name', 'Roll', 'Time']
    for col in required_columns:
        if col not in df.columns:
            # If any column is missing, recreate the file with correct headers
            with open(attendance_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(required_columns)  # Write the correct headers
            df = pd.read_csv(attendance_file)  # Read the freshly created file
            print(f"Warning: Missing column '{col}', file has been fixed.")  # Optional: log the fix
    
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    
    return names, rolls, times, l



# Add Attendance of a specific user
def add_attendance(name, user):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")
    attendance_file = f'Attendance/{user}/Attendance-{datetoday}.csv'

    # Check if attendance file exists
    if os.path.exists(attendance_file):
        df = pd.read_csv(attendance_file)
        # Check if the user ID is already present in the file
        if int(userid) not in list(df['Roll']):
            # Append the new attendance record
            with open(attendance_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([username, userid, current_time])
    else:
        # If file doesn't exist, create it and add attendance
        with open(attendance_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Name', 'Roll', 'Time'])
            writer.writerow([username, userid, current_time])



## A function to get names and rol numbers of all users
def getallusers(user):
    userlist = os.listdir('static/{user}')
    names = []
    rolls = []
    l = len(userlist)

    for i in userlist:
        name, roll = i.split('_')
        names.append(name)
        rolls.append(roll)

    return userlist, names, rolls, l


## A function to delete a user folder 
def deletefolder(duser):
    pics = os.listdir(duser)
    for i in pics:
        os.remove(duser+'/'+i)
    os.rmdir(duser)




################## ROUTING FUNCTIONS #########################

# Our main page
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user_file = 'Users/users.csv'

        if os.path.exists(user_file):
            with open(user_file, mode='r') as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    if row[0] == username and row[1] == password:
                        session['username'] = username
                        return redirect(url_for('index'))

        return render_template('login.html', error="Invalid username or password!")

    return render_template('login.html')


# @app.route('/register', methods=['GET', 'POST'])
# def register():
#     if request.method == 'POST':
#         username = request.form['username']
#         password = request.form['password']

#         # Specify the CSV file location
#         user_file = 'Users/users.csv'

#         # Check if the file exists, and if the username already exists
#         if os.path.exists(user_file):
#             with open(user_file, mode='r') as f:
#                 reader = csv.reader(f)
#                 next(reader)  # Skip the header row
#                 for row in reader:
#                     if row[0] == username:
#                         return render_template('register.html', error="Username already exists. Please choose another one.")

#         # Create the directory and file if they don't exist
#         os.makedirs(os.path.dirname(user_file), exist_ok=True)
#         if not os.path.exists(user_file):
#             with open(user_file, mode='w', newline='') as f:
#                 writer = csv.writer(f)
#                 writer.writerow(['Username', 'Password'])  # Write headers

#         # Append the new user data to the file
#         with open(user_file, mode='a', newline='') as f:
#             writer = csv.writer(f)
#             writer.writerow([username, password])

#         return render_template('login.html', message="Registration successful. Please log in!")

#     return render_template('register.html')

@app.route('/home', methods=['GET', 'POST'])
def home():
    username=session.get('username')
    user_folders = os.listdir(f'Attendance/{username}')  # List of user folders
    slots = {}
    
    # Gather all available slots for each user
    for user in user_folders:
        user_path = os.path.join(f'Attendance/{username}', user)
        if os.path.isdir(user_path):
            slots[user] = [file for file in os.listdir(user_path) if file.endswith('.csv')]
    
    if request.method == 'POST':
        user = request.form.get('user')
        slot = request.form.get('slot')
        slot_file = os.path.join('Attendance', user, slot)

        # Take attendance for the selected slot
        if os.path.exists(slot_file):
            df = pd.read_csv(slot_file)
            username = request.form.get('username')
            current_time = datetime.now().strftime("%H:%M:%S")
            if username not in df['Name'].values:
                with open(slot_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([username, current_time])
                return render_template('home.html', message=f"Attendance taken for {username} in slot {slot}.", slots=slots)

        return render_template('home.html', error="Selected slot or user is invalid.", slots=slots)

    return render_template('home.html', slots=slots)

@app.route('/index')
def index():
    # Check if user is logged in
    if 'username' not in session:
        return redirect(url_for('login'))

    # Get the logged-in user's name
    username = session.get('username')
    attendance_path = f'Attendance/{username}'  # User-specific directory
    slots = []

    # Check if the directory exists and gather slot directories
    if os.path.exists(attendance_path):
        slots = [d for d in os.listdir(attendance_path) if os.path.isdir(os.path.join(attendance_path, d))]

    return render_template('index.html', user=username, slots=slots)


## List users page
@app.route('/listusers')
def listusers():
    user=session.get('username')
    userlist, names, rolls, l = getallusers(user)
    return render_template('listusers.html', userlist=userlist, names=names, rolls=rolls, l=l, totalreg=totalreg(user), datetoday2=datetoday2)


## Delete functionality
@app.route('/deleteuser', methods=['GET'])
def deleteuser():
    user=session.get('username')
    duser = request.args.get('user')
    deletefolder('static/faces/'+duser)

    ## if all the face are deleted, delete the trained file...
    if os.listdir('static/faces/')==[]:
        os.remove('static/face_recognition_model.pkl')
    
    try:
        train_model(user)
    except:
        pass

    userlist, names, rolls, l = getallusers(user)
    return render_template('listusers.html', userlist=userlist, names=names, rolls=rolls, l=l, totalreg=totalreg(user), datetoday2=datetoday2)


# Our main Face Recognition functionality. 
# This function will run when we click on Take Attendance Button.
@app.route('/start', methods=['GET'])
def start():
    user = session.get('username')  # Get the currently logged-in username
    attendance_file = f'Attendance/{user}/Attendance-{datetoday}.csv'
    
    # Ensure the attendance file exists with proper headers
    if not os.path.exists(attendance_file):
        os.makedirs(f'Attendance/{user}', exist_ok=True)  # Create directory if it doesn't exist
        with open(attendance_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Name', 'Roll', 'Time'])  # Write headers for the CSV file
    
    try:
        # Load attendance data and handle missing columns
        df = pd.read_csv(attendance_file)
        if not {'Name', 'Roll', 'Time'}.issubset(df.columns):
            raise KeyError("Attendance file is missing required columns.")
        
        names = df['Name'].tolist()
        rolls = df['Roll'].tolist()
        times = df['Time'].tolist()
        l = len(df)
    except Exception as e:
        # If file is corrupted or missing, reset the data
        names, rolls, times, l = [], [], [], 0
    
    # Check if face recognition model exists
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template(
            'home.html', 
            names=names, rolls=rolls, times=times, l=l, 
            totalreg=totalreg(user), datetoday2=datetoday2, 
            mess='There is no trained model in the static folder. Please add a new face to continue.'
        )
    
    # Start capturing video for face recognition
    cap = cv2.VideoCapture(0)  # Open the default camera
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect and process faces
        faces = extract_faces(frame)
        if len(faces) > 0:
            (x, y, w, h) = faces[0]  # Use the first detected face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (86, 32, 251), 1)  # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y-40), (86, 32, 251), -1)  # Add a filled rectangle for text
            
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))  # Resize the detected face
            identified_person = identify_face(face.reshape(1, -1))[0]  # Predict the identity using the model
            add_attendance(identified_person, user)  # Mark attendance
            
            # Add text label for the identified person
            cv2.putText(frame, f'{identified_person}', (x+5, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Display the video feed
        cv2.imshow('Attendance System', frame)

        # Break the loop on pressing 'q'
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()  # Release the video capture object
    cv2.destroyAllWindows()  # Close all OpenCV windows
    username=session.get('username')
    names, rolls, times, l = extract_attendance(username)
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(username), datetoday2=datetoday2)  # Redirect to the home page



# A function to add a new user.
# This function will run when we add a new user.
@app.route('/add', methods=['GET', 'POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    username=session.get('username')
    userimagefolder = f'static/{username}/'+newusername+'_'+str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    i, j = 0, 0
    cap = cv2.VideoCapture(0)
    while 1:
        _, frame = cap.read()
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
            if j % 5 == 0:
                name = newusername+'_'+str(i)+'.jpg'
                cv2.imwrite(userimagefolder+'/'+name, frame[y:y+h, x:x+w])
                i += 1
            j += 1
        if j == nimgs*5:
            break
        cv2.imshow('Adding new User', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')
    train_model(username)
    names, rolls, times, l = extract_attendance(username)
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(username), datetoday2=datetoday2)



# Our main function which runs the Flask App
if __name__ == "__main__":
    app.run(debug=True)
