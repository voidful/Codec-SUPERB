import React from 'react';
import { useTable, useSortBy } from 'react-table';
import './Leaderboard.css';

const Leaderboard = ({ results }) => {
  const data = React.useMemo(() => {
    return Object.entries(results).map(([key, value]) => ({
      model: key,
      ...Object.keys(value).reduce((acc, curr) => {
        acc[curr] = parseFloat(value[curr].toFixed(3));
        return acc;
      }, {}),
    }));
  }, [results]);

  const columns = React.useMemo(() => {
    const firstItem = results[Object.keys(results)[0]];
    return [
      {
        Header: 'Model',
        accessor: 'model',
      },
      ...Object.keys(firstItem).map(key => ({
        Header: key.toUpperCase(),
        accessor: key,
        sortType: (a, b) => a.original[key] - b.original[key],
      })),
    ];
  }, [results]);

  const {
    getTableProps,
    getTableBodyProps,
    headerGroups,
    rows,
    prepareRow,
  } = useTable({ columns, data }, useSortBy);

  return (
    <table {...getTableProps()} style={{ border: 'solid 1px #333' }}>
      <thead>
        {headerGroups.map(headerGroup => (
          <tr {...headerGroup.getHeaderGroupProps()}>
            {headerGroup.headers.map(column => (
              <th
                {...column.getHeaderProps(column.getSortByToggleProps())}
                style={{
                  borderBottom: 'solid 3px #333',
                  background: '#e0e0e0',
                  color: 'black',
                  fontWeight: 'bold',
                  cursor: 'pointer',
                }}
              >
                {column.render('Header')}
                <span>
                  {column.isSorted ? (column.isSortedDesc ? ' 🔽' : ' 🔼') : ''}
                </span>
              </th>
            ))}
          </tr>
        ))}
      </thead>
      <tbody {...getTableBodyProps()}>
        {rows.map(row => {
          prepareRow(row);
          return (
            <tr {...row.getRowProps()}>
              {row.cells.map(cell => (
                <td
                  {...cell.getCellProps()}
                  style={{
                    padding: '10px',
                    border: 'solid 1px #333',
                    background: '#f9f9f9',
                  }}
                >
                  {cell.render('Cell')}
                </td>
              ))}
            </tr>
          );
        })}
      </tbody>
    </table>
  );
};

export default Leaderboard;
