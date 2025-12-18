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
    const categories = ['Speech', 'Audio', 'Music', 'Overall'];
    const metrics_keys = ['mel', 'pesq', 'stoi', 'f0corr'];

    const colGroups = [
      {
        Header: 'Model Info',
        columns: [
          { Header: 'Model', accessor: 'model' },
          { Header: 'BPS', accessor: 'bps' }
        ]
      }
    ];

    categories.forEach(cat => {
      const catColumns = metrics_keys.map(m => {
        const key = `${cat.toLowerCase()}_${m}`;
        if (key in firstItem || true) { // Force inclusion or check existence
          return {
            Header: m.toUpperCase(),
            accessor: key,
            sortType: 'basic',
          };
        }
        return null;
      }).filter(Boolean);

      if (catColumns.length > 0) {
        colGroups.push({
          Header: cat.toUpperCase(),
          columns: catColumns
        });
      }
    });

    return colGroups;
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
