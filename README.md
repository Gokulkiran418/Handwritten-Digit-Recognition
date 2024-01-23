# Files
## genmodel.py
Run this program first to create the model.

 This is the last line of the code. 
 model.save('G:\\vscode\\Projects\\DigitRecogniton\\mnist_model.h5')

 Change file path to any path you want.

## app.py
The app was created using tkinter

 model = keras.models.load_model('G:\\your\\model\\path\\mnist_model.h5').

 change model path to where you saved your model.

 It takes some time for the app to run. 

Draw the digit slowly and as large as possible then press recognize digit.
If wrong digit is predicted draw it smaller.

the predicted digit is displayed. Click clear canvas to redraw digit

# Prerequisites
Install conda for easier installation of packages like tensorflow.

 
