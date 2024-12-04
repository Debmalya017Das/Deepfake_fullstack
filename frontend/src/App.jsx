import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import VideoUpload from './fu';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<VideoUpload />} />
      </Routes>
    </Router>
  );
}

export default App;
