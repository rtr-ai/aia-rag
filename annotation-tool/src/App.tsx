import './App.scss';
import AnnotationTool from './components/AnnotationTool';
import { store } from './store/store';
import { Provider } from 'react-redux';
function App() {
  return (
    <Provider store={store}>
      <div className="App">
        <AnnotationTool />
      </div>
    </Provider>
  );
}

export default App;
