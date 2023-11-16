import cv2
from goprocam import GoProCamera, constants
from time import time
import socket

gpCam = GoProCamera.GoPro()
sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
t=time()

#gpCam.stream("udp://10.5.5.9:8554",quality="high")
gpCam.livestream("start")
#gpCam.video_settings(res='1080',fps='30')
#gpCam.gpControlSet(constants.Stream.WINDOW_SIZE,constants.Stream.WindowSize.R720)
cap = cv2.VideoCapture("udp://10.5.5.9:8554", cv2.CAP_FFMPEG)



while True:
        nmat , frame = cap.read()
        print(frame.shape)
        cv2.imshow("GoPro OpenCV", frame)
        if cv2.waitKey(1) & 0xFF ==ord('q'):
                break
        if time() - t >=2.5:
                sock.sendto("_GPHD_:0:0:2:0.000000\n".encode(), ('10.5.5.9',8554))
                t=time()
cap.release()
cv2.destroyAllWindows()




exit()








def connect_to_gopro():
    # gopro = GoPro()
    # gopro.open()
    goprocamera = GoProCamera.GoPro(constants.gpcontrol)
    goprocamera.overview()
    goprocamera.video_settings(res='4k', fps='15')

    goprocamera.overview()
    #goprocamera.gpControlSet(constants.Stream.WINDOW_SIZE, constants.Stream.WindowSize.R720)
#    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    #goprocamera.stream()
    


    goprocamera.livestream("start")
    
    
    #Comment lines 89-91 in KeepAlive
    #Change Line 96 to sock.sendto(keep_alive_payload, ("10.5.5.9", 8554))
    
    
    Thread(target=GoProCamera.GoPro.KeepAlive, args=(goprocamera,), daemon=True).start()



connect_to_gopro()




local_url = "udp://127.0.0.1:8554"
cap = cv2.VideoCapture(local_url, cv2.CAP_FFMPEG)





while True:
    

    # Read a frame from the GoPro stream
    ret, frame = cap.read()
    #print(frame.shape)

    if not ret:
        print("Error: Couldn't read frame from the GoPro stream.")
        break

    # Display the frame in a window
    cv2.imshow("GoPro Stream", frame)

    # Press 'q' to exit the loop and close the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture and close the OpenCV window
cap.release()
cv2.destroyAllWindows()

