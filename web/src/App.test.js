import { render, screen } from '@testing-library/react';
import App from './App';

class ResizeObserverMock {
  observe() {}
  unobserve() {}
  disconnect() {}
}

global.ResizeObserver = ResizeObserverMock;

test('renders the Codec-SUPERB paper page', () => {
  render(<App />);
  expect(
    screen.getByRole('heading', {
      name: /Codec-SUPERB: An In-Depth Analysis of Sound Codec Models/i,
    })
  ).toBeInTheDocument();
  expect(screen.getByText(/Latest codec-superb-tiny results/i)).toBeInTheDocument();
  expect(screen.getByText(/BPS\/TPS Analysis/i)).toBeInTheDocument();
  expect(screen.getByText(/LLMCodec Low-Bitrate Strengths/i)).toBeInTheDocument();
  expect(screen.getAllByText(/llmcodec/i).length).toBeGreaterThan(0);
});
