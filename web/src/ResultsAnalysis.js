import React from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import './ResultsAnalysis.css';

const colors = ['#0b5cad', '#2a9d8f', '#e76f51', '#7c3aed', '#64748b', '#c2410c'];
const bpsGroupOrder = ['<1 kbps', '1-3 kbps', '3-8 kbps', '8-16 kbps', '16+ kbps'];

function formatNumber(value, digits = 3) {
  return Number.isFinite(value) ? value.toFixed(digits) : 'N/A';
}

function average(values) {
  const valid = values.filter(Number.isFinite);
  if (!valid.length) {
    return null;
  }
  return valid.reduce((sum, value) => sum + value, 0) / valid.length;
}

function getBpsGroup(bps) {
  if (bps < 1) {
    return '<1 kbps';
  }
  if (bps <= 3) {
    return '1-3 kbps';
  }
  if (bps <= 8) {
    return '3-8 kbps';
  }
  if (bps <= 16) {
    return '8-16 kbps';
  }
  return '16+ kbps';
}

function getTpsGroup(tps) {
  return `${Math.round(tps)} TPS`;
}

function prepareRows(results) {
  return Object.entries(results)
    .filter(([model]) => !model.startsWith('llmcodec_abl_'))
    .map(([model, value]) => ({
      model,
      bps: value.bps,
      tps: value.tps,
      bpsGroup: getBpsGroup(value.bps),
      tpsGroup: getTpsGroup(value.tps),
      overallMel: value.overall_mel,
      overallPesq: value.overall_pesq,
      overallStoi: value.overall_stoi,
      overallF0Corr: value.overall_f0corr,
    }))
    .filter(row => (
      Number.isFinite(row.bps)
      && Number.isFinite(row.tps)
      && Number.isFinite(row.overallMel)
      && Number.isFinite(row.overallPesq)
    ));
}

function summarize(rows, groupKey, order) {
  const grouped = rows.reduce((acc, row) => {
    const key = row[groupKey];
    if (!acc[key]) {
      acc[key] = [];
    }
    acc[key].push(row);
    return acc;
  }, {});

  return Object.entries(grouped)
    .map(([group, groupRows]) => {
      const bestMel = groupRows.reduce((best, row) => (
        !best || row.overallMel < best.overallMel ? row : best
      ), null);
      const bestPesq = groupRows.reduce((best, row) => (
        !best || row.overallPesq > best.overallPesq ? row : best
      ), null);

      return {
        group,
        models: groupRows.length,
        avgMel: average(groupRows.map(row => row.overallMel)),
        avgPesq: average(groupRows.map(row => row.overallPesq)),
        avgStoi: average(groupRows.map(row => row.overallStoi)),
        avgF0Corr: average(groupRows.map(row => row.overallF0Corr)),
        bestMelModel: bestMel?.model,
        bestMelValue: bestMel?.overallMel,
        bestPesqModel: bestPesq?.model,
        bestPesqValue: bestPesq?.overallPesq,
      };
    })
    .sort((a, b) => order(a.group) - order(b.group));
}

function ScatterTooltip({ active, payload }) {
  if (!active || !payload?.length) {
    return null;
  }

  const row = payload[0].payload;
  return (
    <div className="chart-tooltip">
      <strong>{row.model}</strong>
      <span>BPS: {formatNumber(row.bps, 2)}</span>
      <span>TPS: {formatNumber(row.tps, 0)}</span>
      <span>Overall PESQ: {formatNumber(row.overallPesq)}</span>
      <span>Overall MEL: {formatNumber(row.overallMel)}</span>
    </div>
  );
}

function GroupTooltip({ active, payload, label }) {
  if (!active || !payload?.length) {
    return null;
  }

  const row = payload[0].payload;
  return (
    <div className="chart-tooltip">
      <strong>{label}</strong>
      <span>Models: {row.models}</span>
      {payload.map(item => (
        <span key={item.dataKey}>
          {item.name}: {formatNumber(item.value)}
        </span>
      ))}
    </div>
  );
}

const ResultsAnalysis = ({ results }) => {
  const rows = React.useMemo(() => prepareRows(results), [results]);
  const bpsSummary = React.useMemo(() => (
    summarize(rows, 'bpsGroup', group => bpsGroupOrder.indexOf(group))
  ), [rows]);
  const tpsSummary = React.useMemo(() => {
    const orderMap = new Map(
      [...new Set(rows.map(row => row.tpsGroup))]
        .sort((a, b) => Number.parseInt(a, 10) - Number.parseInt(b, 10))
        .map((group, index) => [group, index])
    );
    return summarize(rows, 'tpsGroup', group => orderMap.get(group) ?? 0);
  }, [rows]);
  const tpsSeries = React.useMemo(() => (
    [...new Set(rows.map(row => row.tpsGroup))]
      .sort((a, b) => Number.parseInt(a, 10) - Number.parseInt(b, 10))
      .map((group, index) => ({
        group,
        color: colors[index % colors.length],
        data: rows.filter(row => row.tpsGroup === group),
      }))
  ), [rows]);

  return (
    <section className="analysis-section" aria-labelledby="analysis-title">
      <div className="analysis-heading">
        <p className="section-kicker">BPS/TPS Analysis</p>
        <h2 id="analysis-title">Grouped comparison of bitrate, token rate, and quality</h2>
        <p>
          BPS buckets compare codec bitrate efficiency; TPS groups compare token rate behavior.
          MEL is minimized, while PESQ, STOI, and F0Corr are maximized.
        </p>
      </div>

      <div className="analysis-grid">
        <article className="analysis-panel analysis-wide">
          <div className="panel-heading">
            <h3>Overall PESQ vs BPS</h3>
            <span>Color grouped by TPS</span>
          </div>
          <ResponsiveContainer width="100%" height={340}>
            <ScatterChart margin={{ top: 12, right: 24, bottom: 18, left: 0 }}>
              <CartesianGrid stroke="#d9e0e8" strokeDasharray="4 4" />
              <XAxis
                type="number"
                dataKey="bps"
                name="BPS"
                unit=" kbps"
                tick={{ fill: '#4f5b67', fontSize: 12 }}
                label={{ value: 'BPS (kbps)', position: 'insideBottom', offset: -8, fill: '#4f5b67' }}
              />
              <YAxis
                type="number"
                dataKey="overallPesq"
                name="Overall PESQ"
                tick={{ fill: '#4f5b67', fontSize: 12 }}
                label={{ value: 'Overall PESQ', angle: -90, position: 'insideLeft', fill: '#4f5b67' }}
              />
              <Tooltip content={<ScatterTooltip />} cursor={{ strokeDasharray: '4 4' }} />
              <Legend />
              {tpsSeries.map(series => (
                <Scatter
                  key={series.group}
                  name={series.group}
                  data={series.data}
                  fill={series.color}
                />
              ))}
            </ScatterChart>
          </ResponsiveContainer>
        </article>

        <article className="analysis-panel">
          <div className="panel-heading">
            <h3>Average PESQ by BPS Group</h3>
            <span>Higher is better</span>
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={bpsSummary} margin={{ top: 12, right: 12, bottom: 28, left: 0 }}>
              <CartesianGrid stroke="#d9e0e8" strokeDasharray="4 4" />
              <XAxis dataKey="group" tick={{ fill: '#4f5b67', fontSize: 12 }} />
              <YAxis tick={{ fill: '#4f5b67', fontSize: 12 }} domain={[0, 'auto']} />
              <Tooltip content={<GroupTooltip />} />
              <Bar dataKey="avgPesq" name="Avg PESQ" fill="#0b5cad" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </article>

        <article className="analysis-panel">
          <div className="panel-heading">
            <h3>Average MEL by TPS Group</h3>
            <span>Lower is better</span>
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={tpsSummary} margin={{ top: 12, right: 12, bottom: 28, left: 0 }}>
              <CartesianGrid stroke="#d9e0e8" strokeDasharray="4 4" />
              <XAxis dataKey="group" tick={{ fill: '#4f5b67', fontSize: 12 }} />
              <YAxis tick={{ fill: '#4f5b67', fontSize: 12 }} domain={[0, 'auto']} />
              <Tooltip content={<GroupTooltip />} />
              <Bar dataKey="avgMel" name="Avg MEL" fill="#c2410c" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </article>

        <article className="analysis-panel analysis-wide">
          <div className="panel-heading">
            <h3>Best Models Inside Each BPS Group</h3>
            <span>Comparing best MEL and best PESQ separately</span>
          </div>
          <div className="summary-table-wrap">
            <table className="summary-table">
              <thead>
                <tr>
                  <th>BPS Group</th>
                  <th>Models</th>
                  <th>Best MEL</th>
                  <th>Best PESQ</th>
                  <th>Avg STOI</th>
                  <th>Avg F0Corr</th>
                </tr>
              </thead>
              <tbody>
                {bpsSummary.map(group => (
                  <tr key={group.group}>
                    <td>{group.group}</td>
                    <td>{group.models}</td>
                    <td>{group.bestMelModel} ({formatNumber(group.bestMelValue)})</td>
                    <td>{group.bestPesqModel} ({formatNumber(group.bestPesqValue)})</td>
                    <td>{formatNumber(group.avgStoi)}</td>
                    <td>{formatNumber(group.avgF0Corr)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </article>
      </div>
    </section>
  );
};

export default ResultsAnalysis;
