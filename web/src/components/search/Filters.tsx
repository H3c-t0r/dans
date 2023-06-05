import React from "react";
import { Source } from "./interfaces";
import { getSourceIcon } from "../source";
import { Funnel } from "@phosphor-icons/react";

interface SourceSelectorProps {
  selectedSources: Source[];
  setSelectedSources: React.Dispatch<React.SetStateAction<Source[]>>;
}

const sources: Source[] = [
  { displayName: "Google Drive", internalName: "google_drive" },
  { displayName: "Slack", internalName: "slack" },
  { displayName: "Confluence", internalName: "confluence" },
  { displayName: "Github PRs", internalName: "github" },
  { displayName: "Web", internalName: "web" },
];

export function SourceSelector({
  selectedSources,
  setSelectedSources,
}: SourceSelectorProps) {
  const handleSelect = (source: Source) => {
    setSelectedSources((prev: Source[]) => {
      if (prev.includes(source)) {
        return prev.filter((s) => s.internalName !== source.internalName);
      } else {
        return [...prev, source];
      }
    });
  };

  return (
    <div className="bg-gray-900 p-6">
      <div className="flex mb-3 mx-2">
        <h2 className="font-bold my-auto">Source</h2>
        <Funnel className="my-auto ml-2" size="20" />
      </div>
      {sources.map((source) => (
        <div
          key={source.internalName}
          className={
            "flex cursor-pointer w-full items-center text-white " +
            "py-1.5 my-1.5 rounded-lg px-2 " +
            (selectedSources.includes(source)
              ? "bg-gray-700"
              : "hover:bg-gray-800")
          }
          onClick={() => handleSelect(source)}
        >
          {getSourceIcon(source.internalName, "16")}
          <span className="ml-2 text-sm text-gray-200">
            {source.displayName}
          </span>
        </div>
      ))}
    </div>
  );
}
