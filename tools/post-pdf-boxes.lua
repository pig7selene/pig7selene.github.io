local function escape_latex(text)
  return text
    :gsub("\\", "\\textbackslash{}")
    :gsub("([%%#$&_{}])", "\\%1")
end
local function has_class(element, class_name)
  for _, class in ipairs(element.classes) do
    if class == class_name then
      return true
    end
  end
  return false
end

function Div(element)
  local environment = nil
  local opening = nil

  if has_class(element, "theorem-box") then
    environment = "theorembox"
    opening = "\\begin{theorembox}{" ..
      escape_latex(element.attributes.title or "") .. "}"
  elseif has_class(element, "lemma-box") then
    environment = "lemmabox"
    opening = "\\begin{lemmabox}{" ..
      escape_latex(element.attributes.title or "") .. "}"
  elseif has_class(element, "proof-box") then
    environment = "proofbox"
    opening = "\\begin{proofbox}"
  else
    return nil
  end

  local blocks = { pandoc.RawBlock("latex", opening) }
  for _, block in ipairs(element.content) do
    table.insert(blocks, block)
  end
  table.insert(blocks, pandoc.RawBlock("latex", "\\end{" .. environment .. "}"))
  return blocks
end
