import { useCallback, useEffect, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, glass, metal, paper, plastic, organic, battery, camera } from '../assets';
import Camera from 'react-html5-camera-photo';
import axios from 'axios';

const Form = ({ setImage, setIsPending, setUrl, setColor, setError, setPredict, setAdvice, setTrashBinImage }) => {
  const uploadImage = async (image) => {
    setError(false);
    setIsPending(true);

    const formData = new FormData();
    formData.append('file', image);

    try {
      const response = await axios.post(`http://localhost:8000/predict`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      if (response.status !== 200) {
        throw Error('Internal Server Error');
      }

      const data = response.data;

      setUrl(data.path);
      const adviceResponse = await fetch('http://localhost:8000/get-advice', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ type_trash: data.predicted_value }),
      });

      if (!adviceResponse.ok) {
        throw Error('Failed to fetch advice');
      }

      const adviceData = await adviceResponse.json();
      console.log('Advice:', adviceData.advice);
      console.log('Time:', adviceData.time);

      setPredict(data.predicted_value);
      setAdvice(adviceData.advice);
      switch (data.predicted_value) {
        case 'glass':
          setTrashBinImage(glass);
          setColor('glass');
          break;
        case 'metal':
          setTrashBinImage(metal);
          setColor('metal');
          break;
        case 'paper':
          setTrashBinImage(paper);
          setColor('paper');
          break;
        case 'plastic':
          setTrashBinImage(plastic);
          setColor('plastic');
          break;
        case 'organic':
          setTrashBinImage(organic);
          setColor('organic');
          break;
        case 'battery':
          setTrashBinImage(battery);
          setColor('battery');
          break;
        default:
          setTrashBinImage(null);
          break;
      }
      setIsPending(false);
    } catch (error) {
      console.error('Error:', error);
      setIsPending(false);
      setError(true);
    }
  };

  const [dataUri, setDataUri] = useState('');
  function handleTakePhoto(dataUri) {
    console.log('takePhoto');
    setDataUri(dataUri);
    let byteString = atob(dataUri.split(',')[1]);

    // separate out the mime component
    let mimeString = dataUri.split(',')[0].split(':')[1].split(';')[0];

    let ab = new ArrayBuffer(byteString.length);
    let ia = new Uint8Array(ab);
    for (let i = 0; i < byteString.length; i++) {
      ia[i] = byteString.charCodeAt(i);
    }
    let blob = new Blob([ab], { type: mimeString });
    setImage(URL.createObjectURL(blob));
    uploadImage(blob);
  }

  const onDrop = useCallback(
    async (acceptedFiles) => {
      let file = acceptedFiles[0];
      let reader = new FileReader();

      reader.readAsDataURL(file);
      reader.onload = () => {
        setImage(reader.result);
        uploadImage(file);
      };
    },
    [setImage],
  );
  const handlePaste = (event) => {
    const items = event.clipboardData.items;
    for (let i = 0; i < items.length; i++) {
      const item = items[i];
      if (item.kind === 'file') {
        const blob = item.getAsFile();
        console.log('Pasted file:', blob);
        setImage(URL.createObjectURL(blob));
        uploadImage(blob);
      }
    }
  };
  const { getRootProps, getInputProps, open } = useDropzone({
    onDrop,
    maxFiles: 1,
    accept: { 'image/*': [] },
    noClick: true,
    noKeyboard: true,
  });
  const [isCameraActive, setIsCameraActive] = useState(false);
  const toggleCamera = () => {
    setIsCameraActive((prevState) => !prevState);
  };
  useEffect(() => {
    document.addEventListener('paste', handlePaste);
    return () => {
      document.removeEventListener('paste', handlePaste);
    };
  }, []);
  return (
    <div className="flex flex-col min-h-[50vh] sm:drop-shadow-2xl w-full py-16 sm:px-16 sm:py-10 justify-between bg-white mx-4 sm:mx-0 sm:w-4/6 md:w-3/5 lg:w-fit rounded-3xl">
      <p className="text-center font-semibold text-[1.375rem] sm:text-3xl mt-4 sm:mt-0 mb-4 uppercase text-[#337ea3]">
        garbage classification
      </p>
      <p className="text-center font-thin text-xs text-slate-400 mb-2">File should be Jpeg , Png...</p>
      <div
        {...getRootProps({
          className:
            'h-52 bg-light-grey border-2 border-green-700 border-dashed rounded-2xl flex flex-col justify-center items-center',
        })}
      >
        <input {...getInputProps({ name: 'image' })} />
        <img
          src={Upload}
          className="max-w-1/3 mx-auto mt-2 w-28 sm:w-24"
          draggable="false"
          style={{ userDrag: 'none', filter: 'contrast(15%)' }}
        />
      </div>
      <p className="text-center font-thin text-xs text-slate-400 mt-4 mb-2">Drag & Drop your image here</p>
      <p className="text-center font-thin text-xs text-slate-400 mb-2">Or</p>
      <button
        onClick={open}
        className="bg-[#337ea3] text-gray-900 font-medium p-1 rounded-xl w-auto mx-auto px-4 py-2 text-md hover:bg-[#337ea3] hover:text-white transition-all duration-300"
      >
        Choose a file
      </button>
      <button
        onClick={toggleCamera}
        className="bg-[#337ea3] text-slate-600 font-medium p-1 mt-4 mb-2 rounded-full w-auto mx-auto px-4 py-2 text-md hover:bg-[#337ea3] hover:text-white transition-all duration-300"
      >
        <img src={camera} className="w-6" />
      </button>
      {isCameraActive && (
        <Camera
          onTakePhoto={(dataUri) => {
            handleTakePhoto(dataUri);
          }}
          isFullscreen={false}
          isImageMirror={true}
          idealResolution={{ width: 512, height: 384 }}
          isMaxResolution={false}
          sizeFactor={1}
        />
      )}
    </div>
  );
};

export default Form;
