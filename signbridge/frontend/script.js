document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const video = document.getElementById('webcam');
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    
    const startBtn = document.getElementById('start-btn');
    const stopBtn = document.getElementById('stop-btn');
    const spaceBtn = document.getElementById('space-btn');
    const delBtn = document.getElementById('del-btn');
    const clearBtn = document.getElementById('clear-btn');
    
    const textContainer = document.getElementById('translated-text');
    const connectionDot = document.getElementById('connection-dot');
    const connectionText = document.getElementById('connection-text');
    const videoOverlay = document.getElementById('video-overlay');

    // State Variables
    let currentStream = null;
    let isProcessing = false;
    let backendUrl = 'http://127.0.0.1:5000';
    let fullText = "";
    let lastChar = "";

    function setupCanvas() {
        canvas.width = 640;
        canvas.height = 480;
    }
    
    function updateTextUI(newChar) {
        if (!newChar) return;
        
        if (newChar !== lastChar) {
            fullText += newChar;
            lastChar = newChar;
            renderText();
        }
    }

    function renderText() {
        textContainer.innerHTML = '';
        const span = document.createElement('span');
        span.textContent = fullText;
        textContainer.appendChild(span);
        textContainer.scrollTop = textContainer.scrollHeight;
    }

    // Controls Logic
    spaceBtn.addEventListener('click', () => {
        fullText += " ";
        lastChar = " ";
        renderText();
    });

    delBtn.addEventListener('click', () => {
        if (fullText.length > 0) {
            fullText = fullText.slice(0, -1);
            lastChar = ""; // Reset to allow typing again
            renderText();
        }
    });

    clearBtn.addEventListener('click', () => {
        fullText = "";
        lastChar = "";
        renderText();
    });

    // Start Camera
    startBtn.addEventListener('click', async () => {
        try {
            const response = await fetch(`${backendUrl}/start`, {
                method: 'POST'
            });
            if (!response.ok) throw new Error("Backend not reachable");
            
            currentStream = await navigator.mediaDevices.getUserMedia({ 
                video: { width: 640, height: 480, facingMode: "user" } 
            });
            
            video.srcObject = currentStream;
            setupCanvas();
            
            videoOverlay.style.opacity = '0';
            connectionDot.classList.add('connected');
            connectionText.textContent = 'Connected & Processing';
            
            startBtn.disabled = true;
            stopBtn.disabled = false;
            
            isProcessing = true;
            processFrameLoop();
            
        } catch (err) {
            console.error("Error starting camera/backend:", err);
            alert("Could not connect to camera or backend server. Ensure Flask is running.");
        }
    });

    // Stop Camera
    stopBtn.addEventListener('click', async () => {
        if (currentStream) {
            currentStream.getTracks().forEach(track => track.stop());
            video.srcObject = null;
        }
        
        isProcessing = false;
        
        try {
            await fetch(`${backendUrl}/stop`, { method: 'POST' });
        } catch(e) { console.warn("Backend might already be dead") }
        
        videoOverlay.style.opacity = '1';
        videoOverlay.textContent = "Click 'Start Camera' to begin";
        connectionDot.classList.remove('connected');
        connectionText.textContent = 'Disconnected';
        
        startBtn.disabled = false;
        stopBtn.disabled = true;
    });

    // Video Processing Loop
    async function processFrameLoop() {
        if (!isProcessing) return;
        
        try {
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            const imageData = canvas.toDataURL('image/jpeg', 0.8);
            
            const response = await fetch(`${backendUrl}/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: imageData })
            });
            
            const data = await response.json();
            
            // Re-draw the video frame if canvas is overlaying (clear old landmarks first)
            ctx.clearRect(0, 0, canvas.width, canvas.height); 
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            
            if (data.landmarks && data.landmarks.length > 0) {
                drawLandmarks(data.landmarks);
            }
            
            if (data.prediction && data.prediction !== "Not Trained") {
                updateTextUI(data.prediction);
            }
            
        } catch (error) {
            console.error("Prediction error:", error);
        }
        
        // Schedule next frame (~15 FPS to avoid overwhelming backend)
        setTimeout(processFrameLoop, 66); 
    }
    
    // Helper function to draw connections and points
    function drawLandmarks(landmarks) {
        // MediaPipe connections for hand
        const connections = [
            [0, 1], [1, 2], [2, 3], [3, 4], // Thumb
            [0, 5], [5, 6], [6, 7], [7, 8], // Index
            [5, 9], [9, 10], [10, 11], [11, 12], // Middle
            [9, 13], [13, 14], [14, 15], [15, 16], // Ring
            [13, 17], [17, 18], [18, 19], [19, 20], // Pinky
            [0, 17] // Wrist to pinky base
        ];
        
        ctx.strokeStyle = '#00FF00';
        ctx.lineWidth = 2;
        ctx.fillStyle = '#FF0000';
        
        // Draw connections
        connections.forEach(conn => {
            const startNode = landmarks[conn[0]];
            const endNode = landmarks[conn[1]];
            
            if(startNode && endNode) {
                ctx.beginPath();
                // Landmarks from MediaPipe are normalized [0,1], multiply by width/height
                ctx.moveTo(startNode[0] * canvas.width, startNode[1] * canvas.height);
                ctx.lineTo(endNode[0] * canvas.width, endNode[1] * canvas.height);
                ctx.stroke();
            }
        });
        
        // Draw points
        landmarks.forEach(lm => {
            ctx.beginPath();
            ctx.arc(lm[0] * canvas.width, lm[1] * canvas.height, 4, 0, 2 * Math.PI);
            ctx.fill();
        });
    }
});
