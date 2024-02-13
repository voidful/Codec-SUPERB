import React from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Label, Symbols } from 'recharts';

const EERvsBPSChart = () => {
  const data = [
    { model: 'B1', bps: 2, eer: 4.43 },
    { model: 'B2', bps: 2, eer: 5.22 },
    { model: 'F6', bps: 8, eer: 1.60 }
  ];

  const eData = [
    { model: 'A', bps: 4, eer: 3.31 },
    { model: 'E1', bps: 1.5, eer: 13.58 },
    { model: 'E2', bps: 3, eer: 6.85 },
    { model: 'E3', bps: 6, eer: 4.28 },
  ];

  // 自定義形狀函數
  const renderShape = (props) => {
    const { model } = props.payload;
    const { cx, cy } = props;

    // 根據模型決定形狀
    switch(model) {
      case 'A':
        return <Symbols cx={cx} cy={cy} type="square" size={100} />;
      case 'E1':
        return <Symbols cx={cx} cy={cy} type="triangle" size={100} />;
      case 'E2':
        return <Symbols cx={cx} cy={cy} type="diamond" size={100} />;
      case 'E3':
        return <Symbols cx={cx} cy={cy} type="star" size={100} />;
      default:
        return <circle cx={cx} cy={cy} r={10} fill="green" />;
    }
  };

  return (
    <ResponsiveContainer width="100%" height={400}>
      <ScatterChart
        margin={{ top: 10, right: 10, bottom: 10, left: 10 }}
      >
        <CartesianGrid />
        <XAxis type="number" dataKey="bps" name="Bitrate Per Second" label={{ value: "Bitrate Per Second", position: 'insideBottomRight', offset: -10 }}>
        </XAxis>
        <YAxis type="number" dataKey="eer" name="Equal Error Rate" label={{value: "Equal Error Rate", position: 'insideLeft', offset: 0, angle: -90}}>
        </YAxis>
        <Tooltip cursor={{ strokeDasharray: '3 3' }} />
        <Legend />
        <Scatter name="Models" data={eData} fill="#8884d8" shape="circle" />
        <Scatter name="Encodec" data={data} fill="red" shape={renderShape} />
      </ScatterChart>
    </ResponsiveContainer>
  );
};

export default EERvsBPSChart;
