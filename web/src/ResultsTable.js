import React from 'react';
import { useTable } from 'react-table';
import './ResultsTable.css';

const ResultsTable = ({ dataset, results }) => {
  const data = React.useMemo(() => {
    if (!results[dataset]) return [];
    return Object.entries(results[dataset]).map(([key, value]) => {
      return {
        col1: key,
        ...value,
      };
    });
  }, [dataset, results]);

  const columns = React.useMemo(() => {
    if (!results[dataset]) return [];
    const firstKey = Object.keys(results[dataset])[0];
    return [
      {
        Header: 'Model',
        accessor: 'col1',
      },
      ...Object.keys(results[dataset][firstKey]).map((key) => ({
        Header: key.toUpperCase(),
        accessor: key,
      })),
    ];
  }, [dataset, results]);

  const {
    getTableProps,
    getTableBodyProps,
    headerGroups,
    rows,
    prepareRow,
  } = useTable({ columns, data });

  return (
    <table {...getTableProps()} style={{ border: 'solid 1px #333' }}>
      <thead>
        {headerGroups.map(headerGroup => (
          <tr {...headerGroup.getHeaderGroupProps()}>
            {headerGroup.headers.map(column => (
              <th
                {...column.getHeaderProps()}
                style={{
                  borderBottom: 'solid 3px #333',
                  background: '#e0e0e0',
                  color: 'black',
                  fontWeight: 'bold',
                }}
              >
                {column.render('Header')}
              </th>
            ))}
          </tr>
        ))}
      </thead>
      <tbody {...getTableBodyProps()}>
        {rows.map(row => {
          prepareRow(row)
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
          )
        })}
      </tbody>
    </table>
  );
};

export default ResultsTable;
