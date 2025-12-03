import React from 'react'

function TestButton() {
  const handleClick = () => {
    console.log('Test button clicked!')
    alert('Button works!')
  }

  return (
    <div style={{ padding: '2rem', textAlign: 'center' }}>
      <h3>Test Button</h3>
      <button onClick={handleClick} style={{ padding: '1rem 2rem', fontSize: '1.2rem' }}>
        Click Me to Test
      </button>
    </div>
  )
}

export default TestButton





