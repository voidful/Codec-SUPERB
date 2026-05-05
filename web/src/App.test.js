import { render, screen } from '@testing-library/react';
import App from './App';

test('renders the Codec-SUPERB paper page', () => {
  render(<App />);
  expect(
    screen.getByRole('heading', {
      name: /Codec-SUPERB: An In-Depth Analysis of Sound Codec Models/i,
    })
  ).toBeInTheDocument();
  expect(screen.getByText(/Latest codec-superb-tiny results/i)).toBeInTheDocument();
  expect(screen.getAllByText(/llmcodec_abl_k3/i).length).toBeGreaterThan(0);
});
