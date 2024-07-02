"use client";

import { useContext, useEffect, useRef, useState } from "react";
import { SearchBar } from "./SearchBar";
import { SearchResultsDisplay } from "./SearchResultsDisplay";
import { SourceSelector } from "./filtering/Filters";
import { CCPairBasicInfo, Connector, DocumentSet, Tag, User } from "@/lib/types";
import {
  DanswerDocument,
  Quote,
  SearchResponse,
  FlowType,
  SearchType,
  SearchDefaultOverrides,
  SearchRequestOverrides,
  ValidQuestionResponse,
} from "@/lib/search/interfaces";
import { searchRequestStreamed } from "@/lib/search/streamingQa";
import { SearchHelper } from "./SearchHelper";
import { CancellationToken, cancellable } from "@/lib/search/cancellable";
import { useFilters, useObjectState } from "@/lib/hooks";
import { questionValidationStreamed } from "@/lib/search/streamingQuestionValidation";
import { Persona } from "@/app/admin/assistants/interfaces";
import { PersonaSelector } from "./PersonaSelector";
import { computeAvailableFilters } from "@/lib/filters";
import { useRouter } from "next/navigation";
import { SettingsContext } from "../settings/SettingsProvider";
import { TbLayoutSidebarLeftExpand } from "react-icons/tb";
import { UserDropdown } from "../UserDropdown";
import ResizableSection from "../resizable/ResizableSection";
import { ChatSidebar } from "@/app/chat/sessionSidebar/ChatSidebar";
import { SIDEBAR_WIDTH_CONST } from "@/lib/constants";
import { ChatSession } from "@/app/chat/interfaces";
import { FiBookmark, FiInfo } from "react-icons/fi";
import { HoverPopup } from "../HoverPopup";
import { Logo } from "../Logo";

const SEARCH_DEFAULT_OVERRIDES_START: SearchDefaultOverrides = {
  forceDisplayQA: false,
  offset: 0,
};

const VALID_QUESTION_RESPONSE_DEFAULT: ValidQuestionResponse = {
  reasoning: null,
  answerable: null,
  error: null,
};

interface SearchSectionProps {
  ccPairs: CCPairBasicInfo[];
  documentSets: DocumentSet[];
  personas: Persona[];
  tags: Tag[];

  querySessions: ChatSession[]
  defaultSearchType: SearchType;
  user: User | null
}

export const SearchSection = ({
  ccPairs,
  documentSets,
  personas,
  user,
  tags,
  querySessions,
  defaultSearchType,
}: SearchSectionProps) => {
  // Search Bar
  const [query, setQuery] = useState<string>("");

  // Search
  const [searchResponse, setSearchResponse] = useState<SearchResponse | null>(
    null
  );
  const [isFetching, setIsFetching] = useState(false);

  const [validQuestionResponse, setValidQuestionResponse] =
    useObjectState<ValidQuestionResponse>(VALID_QUESTION_RESPONSE_DEFAULT);

  // Search Type
  const [selectedSearchType, setSelectedSearchType] =
    useState<SearchType>(defaultSearchType);

  const [selectedPersona, setSelectedPersona] = useState<number>(
    personas[0]?.id || 0
  );

  // Filters
  const filterManager = useFilters();
  const availableSources = ccPairs.map((ccPair) => ccPair.source);
  const [finalAvailableSources, finalAvailableDocumentSets] =
    computeAvailableFilters({
      selectedPersona: personas.find(
        (persona) => persona.id === selectedPersona
      ),
      availableSources: availableSources,
      availableDocumentSets: documentSets,
    });

  // Overrides for default behavior that only last a single query
  const [defaultOverrides, setDefaultOverrides] =
    useState<SearchDefaultOverrides>(SEARCH_DEFAULT_OVERRIDES_START);

  // Helpers
  const initialSearchResponse: SearchResponse = {
    answer: null,
    quotes: null,
    documents: null,
    suggestedSearchType: null,
    suggestedFlowType: null,
    selectedDocIndices: null,
    error: null,
    messageId: null,
  };
  const updateCurrentAnswer = (answer: string) =>
    setSearchResponse((prevState) => ({
      ...(prevState || initialSearchResponse),
      answer,
    }));
  const updateQuotes = (quotes: Quote[]) =>
    setSearchResponse((prevState) => ({
      ...(prevState || initialSearchResponse),
      quotes,
    }));
  const updateDocs = (documents: DanswerDocument[]) =>
    setSearchResponse((prevState) => ({
      ...(prevState || initialSearchResponse),
      documents,
    }));
  const updateSuggestedSearchType = (suggestedSearchType: SearchType) =>
    setSearchResponse((prevState) => ({
      ...(prevState || initialSearchResponse),
      suggestedSearchType,
    }));
  const updateSuggestedFlowType = (suggestedFlowType: FlowType) =>
    setSearchResponse((prevState) => ({
      ...(prevState || initialSearchResponse),
      suggestedFlowType,
    }));
  const updateSelectedDocIndices = (docIndices: number[]) =>
    setSearchResponse((prevState) => ({
      ...(prevState || initialSearchResponse),
      selectedDocIndices: docIndices,
    }));
  const updateError = (error: FlowType) =>
    setSearchResponse((prevState) => ({
      ...(prevState || initialSearchResponse),
      error,
    }));
  const updateMessageId = (messageId: number) =>
    setSearchResponse((prevState) => ({
      ...(prevState || initialSearchResponse),
      messageId,
    }));

  let lastSearchCancellationToken = useRef<CancellationToken | null>(null);
  const onSearch = async ({
    searchType,
    offset,
  }: SearchRequestOverrides = {}) => {
    // cancel the prior search if it hasn't finished
    if (lastSearchCancellationToken.current) {
      lastSearchCancellationToken.current.cancel();
    }
    lastSearchCancellationToken.current = new CancellationToken();

    setIsFetching(true);
    setSearchResponse(initialSearchResponse);
    setValidQuestionResponse(VALID_QUESTION_RESPONSE_DEFAULT);
    
    const searchFnArgs = {
      query,
      sources: filterManager.selectedSources,
      documentSets: filterManager.selectedDocumentSets,
      timeRange: filterManager.timeRange,
      tags: filterManager.selectedTags,
      persona: personas.find(
        (persona) => persona.id === selectedPersona
      ) as Persona,
      updateCurrentAnswer: cancellable({
        cancellationToken: lastSearchCancellationToken.current,
        fn: updateCurrentAnswer,
      }),
      updateQuotes: cancellable({
        cancellationToken: lastSearchCancellationToken.current,
        fn: updateQuotes,
      }),
      updateDocs: cancellable({
        cancellationToken: lastSearchCancellationToken.current,
        fn: updateDocs,
      }),
      updateSuggestedSearchType: cancellable({
        cancellationToken: lastSearchCancellationToken.current,
        fn: updateSuggestedSearchType,
      }),
      updateSuggestedFlowType: cancellable({
        cancellationToken: lastSearchCancellationToken.current,
        fn: updateSuggestedFlowType,
      }),
      updateSelectedDocIndices: cancellable({
        cancellationToken: lastSearchCancellationToken.current,
        fn: updateSelectedDocIndices,
      }),
      updateError: cancellable({
        cancellationToken: lastSearchCancellationToken.current,
        fn: updateError,
      }),
      updateMessageId: cancellable({
        cancellationToken: lastSearchCancellationToken.current,
        fn: updateMessageId,
      }),
      selectedSearchType: searchType ?? selectedSearchType,
      offset: offset ?? defaultOverrides.offset,
    };

    const questionValidationArgs = {
      query,
      update: setValidQuestionResponse,
    };

    await Promise.all([
      searchRequestStreamed(searchFnArgs),
      questionValidationStreamed(questionValidationArgs),
    ]);

    setIsFetching(false);
  };

  // handle redirect if search page is disabled
  // NOTE: this must be done here, in a client component since
  // settings are passed in via Context and therefore aren't
  // available in server-side components
  const router = useRouter();
  const settings = useContext(SettingsContext);
  if (settings?.settings?.search_page_enabled === false) {
    router.push("/chat");
  }
  const sidebarElementRef = useRef<HTMLDivElement>(null);
  const innerSidebarElementRef = useRef<HTMLDivElement>(null);


  const [filters, setFilters] = useState(false)
  const toggleFilters = () => {
    setFilters(filters => !filters)
  }

  const [showDocSidebar, setShowDocSidebar] = useState(false)
  const toggleSidebar = () => {
    setShowDocSidebar(showDocSidebar => !showDocSidebar)
  }

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.metaKey || event.ctrlKey) {
        switch (event.key.toLowerCase()) {
          case 'e':
            event.preventDefault();
            toggleSidebar()
            break;

        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [router]);


  const [usedSidebarWidth, setUsedSidebarWidth] = useState<number>(
    300 || parseInt(SIDEBAR_WIDTH_CONST)
  );

  const updateSidebarWidth = (newWidth: number) => {
    setUsedSidebarWidth(newWidth);
    if (sidebarElementRef.current && innerSidebarElementRef.current) {
      sidebarElementRef.current.style.transition = "";
      sidebarElementRef.current.style.width = `${newWidth}px`;
      innerSidebarElementRef.current.style.width = `${newWidth}px`;
    }
  };

  const handleTransitionEnd = (e: React.TransitionEvent<HTMLDivElement>) => {
    if (e.propertyName === 'opacity' && !firstSearch) {
      const target = e.target as HTMLDivElement;
      target.style.display = 'none';
    }
  };
  const [firstSearch, setFirstSearch] = useState(true)


  return (
    <>

      <div
        ref={sidebarElementRef}
        className={`  flex-none absolute left-0 z-[100]  overflow-y-hidden sidebar bg-background-weak h-screen`}
        style={{ width: showDocSidebar ? usedSidebarWidth : 0 }}
      >
        <ResizableSection
          updateSidebarWidth={updateSidebarWidth}
          intialWidth={usedSidebarWidth}
          minWidth={200}
          maxWidth={300 || undefined}
        >
          <div className="w-full relative">

            {/* <ChatSidebar /> */}
            <ChatSidebar
              initialWidth={usedSidebarWidth}
              ref={innerSidebarElementRef}
              closeSidebar={() => toggleSidebar()}
              existingChats={querySessions}
            // currentChatSession={selectedChatSession}
            // folders={folders}
            // openedFolders={openedFolders}
            />
          </div>

        </ResizableSection>

      </div>
      <div className=" left-0 sticky top-0 z-10 w-full bg-opacity-30 backdrop-blur-sm flex">
        <div className="mt-2 flex w-full">
          {!showDocSidebar && (
            <button
              className="ml-4 mt-auto"
              onClick={() => toggleSidebar()}
            >
              <TbLayoutSidebarLeftExpand size={24} />
            </button>
          )}

          <div className="flex mr-4 ml-auto my-auto">
            <UserDropdown user={user} />
          </div>

        </div>
      </div>




      {/* <div className="block 2xl:block w-52 3xl:w-64 mt-4">
        {(ccPairs.length > 0 || documentSets.length > 0) && (
          <SourceSelector
            {...filterManager}
            toggled={filters}
            toggleFilters={toggleFilters}
            availableDocumentSets={finalAvailableDocumentSets}
            existingSources={finalAvailableSources}
            availableTags={tags}
          />
        )}

      </div> */}

      <div className="px-24 pt-10 relative max-w-[2000px] xl:max-w-[1430px] mx-auto">
        <div className="absolute top-24 left-0 hidden 2xl:block w-52 3xl:w-64">
          {(ccPairs.length > 0 || documentSets.length > 0) && (
            <SourceSelector
              {...filterManager}
              toggled={filters}
              toggleFilters={toggleFilters}
              availableDocumentSets={finalAvailableDocumentSets}
              existingSources={finalAvailableSources}
              availableTags={tags}
            />
          )}
        </div>
        <div className="absolute left-0 hidden 2xl:block w-52 3xl:w-64">

        </div>
        <div className="max-w-searchbar-max w-[90%] mx-auto">

          <div
            className={`transition-all duration-500 ease-in-out overflow-hidden ${firstSearch ? 'opacity-100 max-h-[500px]' : 'opacity-0 max-h-0'
              }`}
            onTransitionEnd={handleTransitionEnd}
          >
            <div className="mt-48 mb-8 flex justify-center items-center">
              <div className="w-message-xs 2xl:w-message-sm 3xl:w-message">
                <div className="flex">
                  <div className="mx-auto">
                    <Logo height={80} width={80} className="m-auto" />
                    <div className="m-auto text-3xl font-bold text-strong mt-4 w-fit">
                      Danswer
                    </div>
                    Unlocking your organization's knowledge.
                  </div>
                </div>
              </div>
            </div>
          </div>
          {/* {firstSearch &&
          
            <div className="mt-48 mb-8 flex justify-center items-center h-full">
              <div className="w-message-xs 2xl:w-message-sm 3xl:w-message">
                <div className="flex">
                  <div className="mx-auto">
                    <Logo height={80} width={80} className="m-auto" />
                    <div className="m-auto text-3xl font-bold text-strong mt-4 w-fit">
                      Danswer
                    </div>
                    Unlocking your organization's knowlege.
                  </div>
                </div>
              </div>
            </div>
          } */}

          <SearchBar
            query={query}
            setQuery={setQuery}
            onSearch={async () => {
              setFirstSearch(false)
              setFilters(false)
              setDefaultOverrides(SEARCH_DEFAULT_OVERRIDES_START);
              await onSearch({ offset: 0 });
            }}
          />
          <div className="flex gap-x-4 flex-wrap w-full">

            <div className="block 2xl:block w-52 3xl:w-64 mt-4">

              <div className="pr-5">
                <SearchHelper
                  isFetching={isFetching}
                  searchResponse={searchResponse}
                  selectedSearchType={selectedSearchType}
                  setSelectedSearchType={setSelectedSearchType}
                  defaultOverrides={defaultOverrides}
                  restartSearch={onSearch}
                  forceQADisplay={() =>
                    setDefaultOverrides((prevState) => ({
                      ...(prevState || SEARCH_DEFAULT_OVERRIDES_START),
                      forceDisplayQA: true,
                    }))
                  }
                  setOffset={(offset) => {
                    setDefaultOverrides((prevState) => ({
                      ...(prevState || SEARCH_DEFAULT_OVERRIDES_START),
                      offset,
                    }));
                  }}
                />
              </div>
            </div>
          </div>


          <div className="mt-2">
            <SearchResultsDisplay
              searchResponse={searchResponse}
              validQuestionResponse={validQuestionResponse}
              isFetching={isFetching}
              defaultOverrides={defaultOverrides}
              personaName={
                selectedPersona
                  ? personas.find((p) => p.id === selectedPersona)?.name
                  : null
              }
            />
          </div>
        </div>
      </div >
    </>

  );
};
