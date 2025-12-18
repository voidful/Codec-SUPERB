import React from 'react';
import { useTable, useSortBy } from 'react-table';
import { ChevronUp, ChevronDown, SortAsc } from 'lucide-react';
import './Leaderboard.css';

const Leaderboard = ({ results }) => {
  const data = React.useMemo(() => {
    return Object.entries(results).map(([key, value]) => ({
      model: key,
      ...Object.keys(value).reduce((acc, curr) => {
        acc[curr] = typeof value[curr] === 'number' ? parseFloat(value[curr].toFixed(3)) : value[curr];
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
        sortType: 'basic',
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
    <div className="table-container">
      <table {...getTableProps()}>
        <thead>
          {headerGroups.map(headerGroup => (
            <tr {...headerGroup.getHeaderGroupProps()}>
              {headerGroup.headers.map(column => (
                <th
                  {...column.getHeaderProps(column.getSortByToggleProps())}
                  className={column.isSorted ? 'sorted' : ''}
                >
                  <div className="header-content">
                    {column.render('Header')}
                    <span className="sort-icon">
                      {column.isSorted ? (
                        column.isSortedDesc ? <ChevronDown size={14} /> : <ChevronUp size={14} />
                      ) : (
                        <SortAsc size={14} className="sort-placeholder" />
                      )}
                    </span>
                  </div>
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
                  <td {...cell.getCellProps()}>
                    {cell.column.id === 'model' ? (
                      <span className="model-name">{cell.render('Cell')}</span>
                    ) : (
                      cell.render('Cell')
                    )}
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
};

export default Leaderboard;
