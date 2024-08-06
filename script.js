let video = document.getElementById("vid");
let reactionImage = document.getElementById("reaction");

// Load trained model
let sess = new onnx.InferenceSession();
let loadingModelPromise = sess.loadModel('onnx_model.onnx');

async function setupCamera() {
    return new Promise((resolve, reject) => {
        // Accessing the user camera and video.
        navigator.mediaDevices
            .getUserMedia({
                video: true,
            })
            .then((stream) => {
                // Changing the source of video to current stream.
                video.srcObject = stream;
                video.addEventListener("loadedmetadata", () => {
                    video.play();
                    resolve(video);
                });
            })
            .catch(error => {
                reject(error);
            });
    });
}

// Helper functions for making inference.
function argmax(array) {
    return [].map.call(array, (x, i) => [x, i]).reduce((r, a) => (a[0] > r[0] ? a : r))[1];
}

function indexToReaction(index) {
    const indexDict = {
        0: 'Away from Keyboard',
        1: 'Bye',
        2: 'C',
        3: 'D',
        4: 'E',
        5: 'Perfect',
        6: 'G',
        7: 'H',
        8: 'I',
        9: 'K',
        10: 'Comment',
        11: 'M',
        12: 'N',
        13: 'Stop',
        14: 'P',
        15: 'Question',
        16: 'R',
        17: 'S',
        18: 'Thumbs Up',
        19: 'Remove',
        20: 'V',
        21: 'Yes',
        22: 'No',
        23: 'Y'
    }

    const reactionDict = {
        'Away from Keyboard': 'afk.png',
        'Bye': 'bye.png',
        'Perfect': 'perfect.png',
        'Comment': 'comment.png',
        'Stop': 'stop.png',
        'Question': 'question.png',
        'Yes': 'yes.png',
        'No': 'no.png',
        'Remove': '',
        'Thumbs Up': '1531011081thumbs-up-emoji.png',
    }

    const reaction = indexDict[index];

    return reactionDict[reaction];
}

async function model_init() {
    const model = handPoseDetection.SupportedModels.MediaPipeHands;
    const detectorConfig = {
        runtime: 'mediapipe',
        solutionPath: 'https://cdn.jsdelivr.net/npm/@mediapipe/hands'
        // or 'base/node_modules/@mediapipe/hands' in npm.
    };
    detector = await handPoseDetection.createDetector(model, detectorConfig);

    async function model_inference() {
        const estimationConfig = { flipHorizontal: false };
        const hands = await detector.estimateHands(video, estimationConfig);
        if (hands.length !== 0) {

            const keypoints = hands[0].keypoints3D;

            const coordinates = new Array();

            keypoints.map((coord) => {
                coordinates.push(coord.x);
                coordinates.push(coord.y);
            });

            const input = new onnx.Tensor(new Float32Array(coordinates), 'float32', [1, 42]);

            const outputMap = await sess.run([input]);
            const outputTensor = outputMap.values().next().value;
            const predictions = outputTensor.data;
            //console.log(predictions)
            const index = argmax(predictions);
            const reaction = indexToReaction(index);

            // Display corresponding image
            if (reaction != null) {
                reactionImage.src = reaction;
            }
        }


        requestAnimationFrame(model_inference);
    }

    model_inference();
}



async function main() {
    await setupCamera();
    await model_init();
}

main()