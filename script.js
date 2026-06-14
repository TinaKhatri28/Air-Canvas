import { HandLandmarker, FilesetResolver, DrawingUtils } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/vision_bundle.js";

let handLandmarker = undefined;
let runningMode = "VIDEO";
let enableWebcamButton;
let webcamRunning = false;

// DOM Elements
const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const drawingCanvas = document.getElementById("drawing_canvas");
const drawingCtx = drawingCanvas.getContext("2d");
const currentToolDisplay = document.getElementById("currentToolDisplay");
const colorBoxes = document.querySelectorAll(".color-box");

// App State
const fingerTips = [4, 8, 12, 16, 20];
const colors = ["rgb(0, 0, 255)", "rgb(52, 0, 128)", "rgb(255, 0, 255)", "rgb(0, 0, 0)"];
let colorIndex = 0;
let currentTool = "Draw";
let prevX = 0;
let prevY = 0;
let toolDelay = 20;
let delayCounter = 0;

// Draggable UI Logic
const uiPanel = document.querySelector('.ui-panel');
let isDragging = false;
let currentX, currentY, initialX, initialY;
let xOffset = 0, yOffset = 0;

uiPanel.addEventListener("mousedown", dragStart);
document.addEventListener("mouseup", dragEnd);
document.addEventListener("mousemove", drag);

function dragStart(e) {
    if (e.target.tagName.toLowerCase() === 'button' || e.target.classList.contains('color-box')) {
        return; 
    }
    initialX = e.clientX - xOffset;
    initialY = e.clientY - yOffset;
    isDragging = true;
    uiPanel.style.cursor = 'grabbing';
}

function dragEnd(e) {
    initialX = currentX;
    initialY = currentY;
    isDragging = false;
    uiPanel.style.cursor = 'grab';
}

function drag(e) {
    if (isDragging) {
        e.preventDefault();
        currentX = e.clientX - initialX;
        currentY = e.clientY - initialY;
        xOffset = currentX;
        yOffset = currentY;
        uiPanel.style.transform = `translate3d(${currentX}px, ${currentY}px, 0)`;
    }
}

// Initialize MediaPipe HandLandmarker
const createHandLandmarker = async () => {
    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
    );
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
            delegate: "GPU"
        },
        runningMode: runningMode,
        numHands: 1,
        minHandDetectionConfidence: 0.7,
        minTrackingConfidence: 0.5
    });
};
createHandLandmarker();

// Enable Webcam logic
function hasGetUserMedia() {
    return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

if (hasGetUserMedia()) {
    enableWebcamButton = document.getElementById("enableWebcamButton");
    enableWebcamButton.addEventListener("click", enableCam);
} else {
    console.warn("getUserMedia() is not supported by your browser");
}

function enableCam(event) {
    if (!handLandmarker) {
        console.log("Wait! objectDetector not loaded yet.");
        return;
    }

    if (webcamRunning === true) {
        webcamRunning = false;
        enableWebcamButton.innerText = "Enable Webcam";
    } else {
        webcamRunning = true;
        enableWebcamButton.classList.add("hidden");
    }

    const constraints = {
        video: true
    };

    navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
        video.srcObject = stream;
        video.addEventListener("loadeddata", predictWebcam);
    });
}

function countFingers(landmarks) {
    let fingers = [];
    
    // Thumb
    // Note: To mimic Python x < x-1 for right hand facing camera
    fingers.push(landmarks[fingerTips[0]].x < landmarks[fingerTips[0] - 1].x);
    
    // Other fingers
    for (let i = 1; i < fingerTips.length; i++) {
        let tip = fingerTips[i];
        fingers.push(landmarks[tip].y < landmarks[tip - 2].y);
    }
    return fingers;
}

function arraysEqual(a, b) {
    if (a === b) return true;
    if (a == null || b == null) return false;
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; ++i) {
        if (a[i] !== b[i]) return false;
    }
    return true;
}

function updateUI() {
    currentToolDisplay.textContent = currentTool;
    colorBoxes.forEach((box, index) => {
        if (index === colorIndex) {
            box.classList.add('active');
        } else {
            box.classList.remove('active');
        }
    });
}

let lastVideoTime = -1;
let results = undefined;
const drawingUtils = new DrawingUtils(canvasCtx);

async function predictWebcam() {
    if (canvasElement.width !== video.videoWidth) {
        canvasElement.width = video.videoWidth;
        canvasElement.height = video.videoHeight;
    }
    
    if (drawingCanvas.width !== window.innerWidth || drawingCanvas.height !== window.innerHeight) {
         // Create a temporary canvas to save drawing while resizing
         const tempCanvas = document.createElement('canvas');
         tempCanvas.width = drawingCanvas.width;
         tempCanvas.height = drawingCanvas.height;
         tempCanvas.getContext('2d').drawImage(drawingCanvas, 0, 0);
         
         drawingCanvas.width = window.innerWidth;
         drawingCanvas.height = window.innerHeight;
         
         // Only restore if the old dimensions were valid
         if (tempCanvas.width > 0) {
             drawingCtx.drawImage(tempCanvas, 0, 0);
         }
    }

    let startTimeMs = performance.now();
    if (lastVideoTime !== video.currentTime) {
        lastVideoTime = video.currentTime;
        results = handLandmarker.detectForVideo(video, startTimeMs);
    }

    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    
    // Draw the video frame to output_canvas
    canvasCtx.drawImage(video, 0, 0, canvasElement.width, canvasElement.height);

    if (results.landmarks) {
        for (const landmarks of results.landmarks) {
            // Draw Hand Landmarks
            drawingUtils.drawConnectors(landmarks, HandLandmarker.HAND_CONNECTIONS, {
                color: "#00FF00",
                lineWidth: 2
            });
            drawingUtils.drawLandmarks(landmarks, { color: "#FF0000", lineWidth: 1, radius: 3 });

            const fingers = countFingers(landmarks);
            
            const indexFinger = landmarks[8];
            const x = Math.floor(indexFinger.x * drawingCanvas.width);
            const y = Math.floor(indexFinger.y * drawingCanvas.height);

            if (delayCounter === 0) {
                if (arraysEqual(fingers, [true, true, false, false, false])) {
                    currentTool = "Switch";
                    delayCounter = toolDelay;
                } else if (arraysEqual(fingers, [false, true, false, false, false])) {
                    currentTool = "Draw";
                } else if (arraysEqual(fingers, [true, true, true, true, true])) {
                    currentTool = "Change color";
                    colorIndex = (colorIndex + 1) % colors.length;
                    delayCounter = toolDelay;
                } else if (arraysEqual(fingers, [true, false, false, false, false])) {
                    currentTool = "Erase";
                    delayCounter = toolDelay;
                } else if (arraysEqual(fingers, [false, true, true, true, false])) {
                    currentTool = "Cleared";
                    // Clear the drawing canvas
                    drawingCtx.clearRect(0, 0, drawingCanvas.width, drawingCanvas.height);
                    delayCounter = toolDelay;
                }
                updateUI();
            }

            if (currentTool === "Draw" && fingers[1]) {
                if (prevX === 0 && prevY === 0) {
                    prevX = x;
                    prevY = y;
                }
                drawingCtx.beginPath();
                drawingCtx.moveTo(prevX, prevY);
                drawingCtx.lineTo(x, y);
                drawingCtx.strokeStyle = colors[colorIndex];
                drawingCtx.lineWidth = 5;
                drawingCtx.lineCap = 'round';
                drawingCtx.stroke();
                prevX = x;
                prevY = y;
            } else if (currentTool === "Erase" && fingers[0]) {
                if (prevX === 0 && prevY === 0) {
                    prevX = x;
                    prevY = y;
                }
                // To erase on a transparent canvas, we use destination-out
                drawingCtx.globalCompositeOperation = "destination-out";
                drawingCtx.beginPath();
                drawingCtx.moveTo(prevX, prevY);
                drawingCtx.lineTo(x, y);
                drawingCtx.lineWidth = 30;
                drawingCtx.lineCap = 'round';
                drawingCtx.stroke();
                // Restore composite operation
                drawingCtx.globalCompositeOperation = "source-over";
                
                prevX = x;
                prevY = y;
            } else {
                prevX = 0;
                prevY = 0;
            }
        }
    } else {
        prevX = 0;
        prevY = 0;
    }

    if (delayCounter > 0) {
        delayCounter -= 1;
    }

    canvasCtx.restore();

    if (webcamRunning === true) {
        window.requestAnimationFrame(predictWebcam);
    }
}
