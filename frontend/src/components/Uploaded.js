//import CheckIcon from '@mui/icons-material/Check';

const Uploaded = ({ image, predict, advice, color, TrashBinImage }) => {
  const handleRetry = () => {
    window.location.reload();
  };
  return (
    <>
      <div className="fixed left-0 top-1/2 -translate-y-1/2 sm:drop-shadow-lg gap-6 py-16 items-center justify-center w-2/5 mx-4 bg-white md:w-3/5 rounded-3xl transition-all duration-500 transform translate-x- sm:translate-x-40 hidden sm:block">
        <div className="flex items-center">
          <img src={TrashBinImage} alt="Trash Bin" className="h-16" />
          <p className={`ml-0 uppercase text font-bold ${color}`}>{predict}</p>
        </div>
      </div>
      <div className="flex flex-col sm:drop-shadow-lg gap-6 py-16 items-center justify-center w-full mx-4 bg-white md:w-1/2 rounded-3xl">
        <p className="text-center font-semibold text-2xl uppercase text-[#337ea3]">garbage classification</p>
        <div>
          <img src={image} className="max-w-full mx-auto h-[28vh] sm:h-[28vh] rounded-3xl" />
        </div>
        <div className="flex justify-center">
          <div className="bg-[#337ea3]/40 text-slate-600 text-justify text-md rounded-xl px-8 py-3 mx-12">
            {advice.split('\n').map((item, i) => (
              <p key={i} className="mt-2">
                {item}
              </p>
            ))}
          </div>
        </div>
        <button
          onClick={handleRetry}
          className="bg-[#337ea3]/40 text-slate-600 font-medium rounded-xl w-auto mx-auto px-6 py-3 text-md hover:bg-[#337ea3]/75 hover:text-white transition-all duration-300"
        >
          Retry
        </button>
      </div>
    </>
  );
};

export default Uploaded;
