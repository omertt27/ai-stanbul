import React from 'react';

const SearchBar = ({ value, onChange, onSubmit }) => (
  <form onSubmit={onSubmit} className="flex items-center gap-2">
    <input
      type="text"
      value={value}
      onChange={onChange}
      placeholder="Search..."
      className="border rounded px-3 py-2 focus:outline-none focus:ring"
    />
    <button type="submit" className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 transition">
      Search
    </button>
  </form>
);

export default SearchBar;
