import "./App.css";
import { styled } from "@mui/material/styles";
import Button from "@mui/material/Button";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import SendIcon from "@mui/icons-material/Send";
import { Typography } from "@mui/material";
import { useEffect, useRef, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import * as Tone from 'tone'
import React from "react";

const VisuallyHiddenInput = styled("input")({
  clip: "rect(0 0 0 0)",
  clipPath: "inset(50%)",
  height: 1,
  overflow: "hidden",
  position: "absolute",
  bottom: 0,
  left: 0,
  whiteSpace: "nowrap",
  width: 1,
});

function App() {
  const [arrayBuffer, setArrayBuffer] = useState(null);
  const [audioBuffer, setAudioBuffer] = useState(null);
  const [features, setFeatures] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const audioContextRef = useRef(null);
  const modelRef = useRef(null);

  const handleFileChange = (event) => {
    const uploadedFile = event.target.files[0];
    if (!uploadedFile) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      setArrayBuffer(event.target.result);
    };
    reader.readAsArrayBuffer(uploadedFile);
    decodeAudio();
  };

  const extractFeatures = (audioBuffer) => {
    // 1. Convert the audio buffer to a single channel (if needed)
    const channelData = audioBuffer.getChannelData(0); // Assuming single channel

    // 2. Create a FFT (Fast Fourier Transform) object
    const fft = new Tone.FFT(channelData.length);

    // 3. Analyze the audio data using the FFT
    fft.realToFreq(channelData);
    const spectrum = fft.getSpectrum(); // Get the frequency spectrum

    // 4. Create a Mel-frequency filter bank
    const melFilterBank = Tone.getMelFilterBank(
      spectrum.length,
      audioBuffer.sampleRate,
      40,
      500,
      15000
    ); // Parameters may need adjustment

    // 5. Apply the filter bank to the spectrum and take the log
    const melEnergies = melFilterBank
      .process(spectrum)
      .map((value) => Math.log(value + 1.0));

    // 6. Perform a Discrete Cosine Transform (DCT) on the mel energies
    const dct = new Tone.DCT(melEnergies.length);
    const mfccs = dct.process(melEnergies);

    // 7. Return the extracted MFCCs
    return mfccs;
  };

  const predict = async () => {
    if (!features || !modelRef.current) return;

    const featuresTensor = tf.tensor(features);
    const predictions = await modelRef.current.predict(featuresTensor);
    setPredictions(predictions.dataSync());
  };

  const decodeAudio = async () => {
    const audioContext = audioContextRef.current;
    if (!audioContext) return;

    try {
      const decodedBuffer = await audioContext.decodeAudioData(arrayBuffer);
      setAudioBuffer(decodedBuffer);
    } catch (error) {
      console.error("Error decoding audio data:", error);
    }
  };

  useEffect(() => {
    // Load the model on component mount
    tf.loadLayersModel("model/model.json")
      .then((model) => {
        modelRef.current = model;
      })
      .catch((error) => {
        console.error("Error loading model:", error);
      });

    return () => {
      // Clean up resources (optional)
    };
  }, []);

  useEffect(() => {
    if (audioBuffer) {
      const extractedFeatures = extractFeatures(audioBuffer);
      setFeatures(extractedFeatures);
    }
  }, [audioBuffer]);

  useEffect(() => {
    audioContextRef.current = new AudioContext();
    // Clean up the audio context when the component unmounts
    return () => {
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
    };
  }, []);

  return (
    <div className="App">
      <Typography variant="h1" gutterBottom>
        Upload Songs
      </Typography>
      <Typography variant="h5" gutterBottom>
        Upload up to five songs to start
      </Typography>

      <Button
        component="label"
        role={undefined}
        variant="contained"
        tabIndex={-1}
        startIcon={<CloudUploadIcon />}
      >
        Upload file
        <VisuallyHiddenInput
          type="file"
          accept="audio/mpeg, audio/ogg, audio/*"
          multiple
          onChange={handleFileChange}
        />
        {audioBuffer && "done"}
      </Button>

      {features && <button onClick={predict}>Classify Audio</button>}
      {predictions && (
        <p>
          Predictions:{" "}
          {predictions.map((value, index) => (
            <span key={index}>
              {index}: {Math.round(value * 100)}%
            </span>
          ))}
        </p>
      )}
      <Typography variant="h1" gutterBottom>
        Singers
      </Typography>
      <Typography variant="h5" gutterBottom>
        select a target singer
      </Typography>
      <Button variant="contained" endIcon={<SendIcon />}>
        Generate Cover
      </Button>
      <Button variant="contained" endIcon={<SendIcon />}>
        Recommendations
      </Button>
    </div>
  );
}

export default App;
