import React, { useCallback } from "react";

interface SelectOption {
  label: string;
  value: string | number;
}

interface SelectProps {
  value: string[];
  onChange: (value: string[]) => void;
  options: SelectOption[];
  mode: "tags" | "multiple";
  style?: React.CSSProperties;
}

const Select: React.FC<SelectProps> = React.memo(({ value, onChange, options, mode, style }) => {
  const [inputValue, setInputValue] = React.useState<string>("");
  const [selectedOptions, setSelectedOptions] = React.useState<string[]>(
    value || []
  );
  const [isDropdownVisible, setIsDropdownVisible] =
    React.useState<boolean>(false);

    React.useEffect(() => {
      setSelectedOptions((prev) => (prev !== value ? value : prev));
    }, [value]);
    
  const handleInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setInputValue(event.target.value);
    setIsDropdownVisible(true);
  };

  const filteredOptions = React.useMemo(() => {
    const lowerInput = inputValue.toLowerCase();
    return options.filter(
      (option) =>
        !selectedOptions.includes(option.value.toString()) &&
        option.label.toLowerCase().includes(lowerInput)
    );
  }, [inputValue, options, selectedOptions]);

  const handleAddTag = () => {
    if (mode === "multiple") {
      return;
    }
    if (inputValue.trim() && !selectedOptions.includes(inputValue)) {
      const updatedSelectedOptions = [...selectedOptions, inputValue];
      setSelectedOptions(updatedSelectedOptions);
      onChange(updatedSelectedOptions);
      setInputValue("");
    }
    setIsDropdownVisible(false);
  };

  const handleRemoveTag = useCallback((optionValue: string) => {
    setSelectedOptions((prev) => {
      const updated = prev.filter((val) => val !== optionValue);
      onChange(updated);
      return updated;
    });
  }, [onChange]);
  
  const handleOptionSelect = useCallback((optionValue: string) => {
    setSelectedOptions((prev) => {
      if (!prev.includes(optionValue)) {
        const updated = [...prev, optionValue];
        onChange(updated);
        return updated;
      }
      return prev;
    });
    setIsDropdownVisible(false);
  }, [onChange]);

  return (
    <div className="uk-form-controls uk-margin-bottom" style={style}>
      <div>
        <div className="uk-flex uk-flex-middle uk-flex-wrap row-gap-md">
          {selectedOptions.map((selected) => (
            <span
              key={selected}
              className="uk-label uk-margin-small-right"
              style={{ display: "flex", alignItems: "center" }}
            >
              {options.find((p) => p.value.toString() === selected)?.label ||
                selected}
              <button
                type="button"
                className="uk-button uk-button-link uk-margin-small-left"
                style={{
                  color: "#ffffff",
                }}
                onClick={() => handleRemoveTag(selected)}
              >
                ✕
              </button>
            </span>
          ))}
        </div>
      </div>

      <div
        className="uk-flex uk-flex-middle uk-margin-top"
        style={{ position: "relative" }}
      >
        <input
          type="text"
          className="uk-input"
          placeholder="Eintippen oder auswählen"
          value={inputValue}
          onFocus={() => setIsDropdownVisible(true)}
          onBlur={() => setTimeout(() => setIsDropdownVisible(false), 200)}
          onChange={handleInputChange}
          onKeyDown={(e) => {
            if (e.key === "Enter" && mode === "tags") {
              e.preventDefault();
              handleAddTag();
            }
          }}
        />
        {isDropdownVisible && (
          <ul
            className="uk-list uk-list-divider"
            style={{
              position: "absolute",
              top: "100%",
              left: 0,
              right: 0,
              marginTop: "5px!important",
              maxHeight: "150px",
              overflowY: "auto",
              background: "white",
              border: "1px solid #ccc",
              zIndex: 1000,
            }}
          >
            {inputValue.length > 0 &&
              mode === "tags" &&
              !selectedOptions.includes(inputValue) && (
                <li
                  key={inputValue}
                  className="uk-flex uk-flex-between uk-flex-middle uk-flex-wrap"
                  style={{ cursor: "pointer", padding: "5px" }}
                  onClick={() => handleOptionSelect(inputValue)}
                >
                  {inputValue}
                </li>
              )}
            {filteredOptions.map((option) => (
                <li
                  key={option.value}
                  className="uk-flex uk-flex-between uk-flex-middle uk-flex-wrap"
                  style={{ cursor: "pointer", padding: "5px" }}
                  onClick={() => handleOptionSelect(option.value.toString())}
                >
                  {option.label || option.value}
                </li>
              ))}
          </ul>
        )}
      </div>
    </div>
  );
});

export { Select };
