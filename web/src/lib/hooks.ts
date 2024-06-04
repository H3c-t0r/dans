import {
  ConnectorIndexingStatus,
  Credential,
  DocumentBoostStatus,
  Tag,
  User,
  UserGroup,
} from "@/lib/types";
import useSWR, { mutate, useSWRConfig } from "swr";
import { errorHandlingFetcher } from "./fetcher";
import { RefObject, useState } from "react";
import { DateRangePickerValue } from "@tremor/react";
import { SourceMetadata } from "./search/interfaces";
import { EE_ENABLED } from "./constants";

const CREDENTIAL_URL = "/api/manage/admin/credential";

import { useEffect, useRef } from "react";

export type AutoScrollHookType = {
  isStreaming: boolean;
  lastMessageRef: RefObject<HTMLDivElement>;
  inputRef: RefObject<HTMLDivElement>;
  endDivRef: RefObject<HTMLDivElement>;
  distance?: number;
};

export const useAutoScrollOnMessage = ({
  isStreaming,
  lastMessageRef,
  inputRef,
  endDivRef,
  distance = 140,
}: AutoScrollHookType) => {
  useEffect(() => {
    if (isStreaming && lastMessageRef.current && inputRef.current) {
      const lastMessageRect = lastMessageRef.current.getBoundingClientRect();
      const endDivRect = inputRef.current.getBoundingClientRect();

      if (
        endDivRect.bottom - lastMessageRect.bottom > distance &&
        endDivRef?.current
      ) {
        endDivRef.current.scrollIntoView({ behavior: "smooth" });
      }
    }
  });
};

export type InitialScrollType = {
  isFetchingChatMessages: boolean;
  endDivRef: RefObject<HTMLDivElement>;
  hasPerformedInitialScroll: boolean;
  initialScrollComplete: () => void;
  isStreaming: boolean;
};

export const useInitialScroll = ({
  isStreaming,
  endDivRef,
  isFetchingChatMessages,
  hasPerformedInitialScroll,
  initialScrollComplete,
}: InitialScrollType) => {
  useEffect(() => {
    if (!hasPerformedInitialScroll && endDivRef.current && isStreaming) {
      endDivRef.current.scrollIntoView({ behavior: "smooth" });
      initialScrollComplete();
    }
  }),
    [isFetchingChatMessages];
};

export type ResponsiveScrollType = {
  lastMessageRef: RefObject<HTMLDivElement>;
  inputRef: RefObject<HTMLDivElement>;
  endDivRef: RefObject<HTMLDivElement>;
  textAreaRef: RefObject<HTMLTextAreaElement>;
};

export const useResponsiveAutoScroll = ({
  lastMessageRef,
  inputRef,
  endDivRef,
  textAreaRef,
}: ResponsiveScrollType) => {
  useEffect(() => {
    let timeoutId: NodeJS.Timeout | null = null;
    let prevInputHeight = 0;
    let prevDistance = 0;

    const handleInputResize = async () => {
      function delay(ms: number) {
        return new Promise((resolve) => setTimeout(resolve, ms));
      }
      await delay(100);

      // validate that message and input exist
      if (
        lastMessageRef &&
        inputRef &&
        inputRef.current &&
        lastMessageRef.current
      ) {
        const lastMessageRect = lastMessageRef.current.getBoundingClientRect();
        const endDivRect = inputRef.current.getBoundingClientRect();
        const currentInputHeight = endDivRect.height;

        if (
          // validate change in height
          currentInputHeight > prevInputHeight &&
          // validate distance
          endDivRect.top <= lastMessageRect.bottom
        ) {
          // Validate previous distance and existence of the final div
          if (prevDistance > -100 && endDivRef && endDivRef?.current) {
            if (timeoutId) {
              clearTimeout(timeoutId);
            }

            timeoutId = setTimeout(() => {
              if (endDivRef && endDivRef?.current) {
                // TODO
                // window.scrollBy({
                //   top: currentInputHeight - prevInputHeight,
                //   behavior: 'smooth',
                // });

                endDivRef?.current.scrollIntoView({ behavior: "smooth" });
              }
            }, 500);
          }

          prevInputHeight = currentInputHeight;
        }
        if (currentInputHeight !== prevInputHeight) {
          prevInputHeight = currentInputHeight;
        }
      }

      // Update previous distance for calculation (regardless of scroll)
      if (
        lastMessageRef &&
        inputRef &&
        inputRef.current &&
        lastMessageRef.current
      ) {
        const lastMessageRect = lastMessageRef.current.getBoundingClientRect();
        const endDivRect = inputRef.current.getBoundingClientRect();
        prevDistance = endDivRect.top - lastMessageRect.bottom;
      }
    };

    // Previous- on input
    const textarea = textAreaRef.current;
    if (textarea) {
      textarea.addEventListener("input", handleInputResize);
    }

    return () => {
      if (textarea) {
        textarea.removeEventListener("input", handleInputResize);
      }
      if (timeoutId) {
        clearTimeout(timeoutId);
      }
    };
  }, [lastMessageRef, inputRef]);

  return null;
};

export const usePublicCredentials = () => {
  const { mutate } = useSWRConfig();
  const swrResponse = useSWR<Credential<any>[]>(
    CREDENTIAL_URL,
    errorHandlingFetcher
  );

  return {
    ...swrResponse,
    refreshCredentials: () => mutate(CREDENTIAL_URL),
  };
};

const buildReactedDocsUrl = (ascending: boolean, limit: number) => {
  return `/api/manage/admin/doc-boosts?ascending=${ascending}&limit=${limit}`;
};

export const useMostReactedToDocuments = (
  ascending: boolean,
  limit: number
) => {
  const url = buildReactedDocsUrl(ascending, limit);
  const swrResponse = useSWR<DocumentBoostStatus[]>(url, errorHandlingFetcher);

  return {
    ...swrResponse,
    refreshDocs: () => mutate(url),
  };
};

export const useObjectState = <T>(
  initialValue: T
): [T, (update: Partial<T>) => void] => {
  const [state, setState] = useState<T>(initialValue);
  const set = (update: Partial<T>) => {
    setState((prevState) => {
      return {
        ...prevState,
        ...update,
      };
    });
  };
  return [state, set];
};

const INDEXING_STATUS_URL = "/api/manage/admin/connector/indexing-status";

export const useConnectorCredentialIndexingStatus = (
  refreshInterval = 30000 // 30 seconds
) => {
  const { mutate } = useSWRConfig();
  const swrResponse = useSWR<ConnectorIndexingStatus<any, any>[]>(
    INDEXING_STATUS_URL,
    errorHandlingFetcher,
    { refreshInterval: refreshInterval }
  );

  return {
    ...swrResponse,
    refreshIndexingStatus: () => mutate(INDEXING_STATUS_URL),
  };
};

export const useTimeRange = (initialValue?: DateRangePickerValue) => {
  return useState<DateRangePickerValue | null>(null);
};

export interface FilterManager {
  timeRange: DateRangePickerValue | null;
  setTimeRange: React.Dispatch<
    React.SetStateAction<DateRangePickerValue | null>
  >;
  selectedSources: SourceMetadata[];
  setSelectedSources: React.Dispatch<React.SetStateAction<SourceMetadata[]>>;
  selectedDocumentSets: string[];
  setSelectedDocumentSets: React.Dispatch<React.SetStateAction<string[]>>;
  selectedTags: Tag[];
  setSelectedTags: React.Dispatch<React.SetStateAction<Tag[]>>;
}

export function useFilters(): FilterManager {
  const [timeRange, setTimeRange] = useTimeRange();
  const [selectedSources, setSelectedSources] = useState<SourceMetadata[]>([]);
  const [selectedDocumentSets, setSelectedDocumentSets] = useState<string[]>(
    []
  );
  const [selectedTags, setSelectedTags] = useState<Tag[]>([]);

  return {
    timeRange,
    setTimeRange,
    selectedSources,
    setSelectedSources,
    selectedDocumentSets,
    setSelectedDocumentSets,
    selectedTags,
    setSelectedTags,
  };
}

export const useUsers = () => {
  const url = "/api/manage/users";
  const swrResponse = useSWR<User[]>(url, errorHandlingFetcher);

  return {
    ...swrResponse,
    refreshIndexingStatus: () => mutate(url),
  };
};

export interface LlmOverride {
  name: string;
  provider: string;
  modelName: string;
}

export interface LlmOverrideManager {
  llmOverride: LlmOverride;
  setLlmOverride: React.Dispatch<React.SetStateAction<LlmOverride>>;
  temperature: number | null;
  setTemperature: React.Dispatch<React.SetStateAction<number | null>>;
}

export function useLlmOverride(): LlmOverrideManager {
  const [llmOverride, setLlmOverride] = useState<LlmOverride>({
    name: "",
    provider: "",
    modelName: "",
  });
  const [temperature, setTemperature] = useState<number | null>(null);

  return {
    llmOverride,
    setLlmOverride,
    temperature,
    setTemperature,
  };
}

/* 
EE Only APIs
*/

const USER_GROUP_URL = "/api/manage/admin/user-group";

export const useUserGroups = (): {
  data: UserGroup[] | undefined;
  isLoading: boolean;
  error: string;
  refreshUserGroups: () => void;
} => {
  const swrResponse = useSWR<UserGroup[]>(USER_GROUP_URL, errorHandlingFetcher);

  if (!EE_ENABLED) {
    return {
      ...{
        data: [],
        isLoading: false,
        error: "",
      },
      refreshUserGroups: () => {},
    };
  }

  return {
    ...swrResponse,
    refreshUserGroups: () => mutate(USER_GROUP_URL),
  };
};
