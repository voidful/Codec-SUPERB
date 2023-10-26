import React from 'react'
import { useTable } from 'react-table'
import './ResultsTable.css'

const ResultsTable = ({ results }) => {
  const data = React.useMemo(() => {
    return Object.entries(results.librispeech_asr_dummy).map(([key, value]) => {
      return {
        col1: key,
        ...value,
      }
    })
  }, [results])

  const columns = React.useMemo(() => {
    return [
      {
        Header: 'Model',
        accessor: 'col1', // accessor is the "key" in the data
      },
      ...Object.keys(results.librispeech_asr_dummy.descript_audio_codec).map(key => ({
        Header: key.toUpperCase(),
        accessor: key,
      })),
    ]
  }, [results])

  const {
    getTableProps,
    getTableBodyProps,
    headerGroups,
    rows,
    prepareRow,
  } = useTable({ columns, data })

  return (
    <table {...getTableProps()} style={{ border: 'solid 1px blue' }}>
      <thead>
        {headerGroups.map(headerGroup => (
          <tr {...headerGroup.getHeaderGroupProps()}>
            {headerGroup.headers.map(column => (
              <th
                {...column.getHeaderProps()}
                style={{
                  borderBottom: 'solid 3px red',
                  background: 'aliceblue',
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
                    border: 'solid 1px gray',
                    background: 'papayawhip',
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
  )
}

export default ResultsTable
