const fs = require('fs');
const path = require('path');

const resultsDir = path.join(__dirname, 'results');
const output = [];

fs.readdirSync(resultsDir).forEach(file => {
  if (file.endsWith('.json')) {
    output.push(file.replace('.json', ''));
  }
});

fs.writeFileSync(
  path.join(__dirname, 'results', 'index.js'),
  `export default ${JSON.stringify(output)};`
);
