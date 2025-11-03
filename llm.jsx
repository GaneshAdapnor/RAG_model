import React, { useState, useRef, useEffect } from 'react';
import { Upload, Send, FileText, Trash2, Loader2, AlertCircle, RefreshCw } from 'lucide-react';

const RAGSystem = () => {
  const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
  
  const [documents, setDocuments] = useState([]);
  const [question, setQuestion] = useState('');
  const [messages, setMessages] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState('');
  const fileInputRef = useRef(null);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    // Load documents from API on mount
    loadDocuments();
  }, []);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const loadDocuments = async () => {
    try {
      const response = await fetch(`${API_URL}/api/documents`);
      if (!response.ok) throw new Error('Failed to load documents');
      
      const data = await response.json();
      setDocuments(data.documents || []);
    } catch (err) {
      console.error('Error loading documents:', err);
      // Don't show error on initial load if API is not available
    }
  };

  const handleFileUpload = async (e) => {
    const files = Array.from(e.target.files);
    if (files.length === 0) return;

    setIsProcessing(true);
    setError('');

    try {
      const uploadPromises = files.map(async (file) => {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('auto_summarize', 'true');

        const response = await fetch(`${API_URL}/api/upload`, {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({ detail: response.statusText }));
          throw new Error(errorData.detail || `Failed to upload ${file.name}`);
        }

        return await response.json();
      });

      const results = await Promise.allSettled(uploadPromises);
      const successful = [];
      const failed = [];

      results.forEach((result, index) => {
        if (result.status === 'fulfilled') {
          successful.push(result.value);
        } else {
          failed.push({ file: files[index].name, error: result.reason.message });
        }
      });

      if (failed.length > 0) {
        failed.forEach(({ file, error }) => {
          setMessages(prev => [...prev, {
            type: 'system',
            content: `⚠️ Error processing ${file}: ${error}`
          }]);
        });
      }

      if (successful.length > 0) {
        // Reload documents list
        await loadDocuments();
        
        successful.forEach((doc) => {
          setMessages(prev => [...prev, {
            type: 'system',
            content: `✅ Successfully uploaded ${doc.filename}. Created ${doc.chunks_created} chunks.${doc.summary ? '\n\nSummary: ' + doc.summary : ''}`
          }]);
        });
      }

      if (successful.length === 0 && failed.length > 0) {
        setError('No documents could be uploaded. Please check the file formats and try again.');
      }
    } catch (err) {
      console.error('Upload error:', err);
      setError(`Error: ${err.message}`);
    } finally {
      setIsProcessing(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };


  const handleAskQuestion = async () => {
    if (!question.trim()) return;
    if (documents.length === 0) {
      setError('Please upload documents first');
      return;
    }

    setIsProcessing(true);
    setError('');

    const userMessage = { type: 'user', content: question };
    setMessages(prev => [...prev, userMessage]);
    const questionText = question;
    setQuestion('');

    try {
      const response = await fetch(`${API_URL}/api/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question: questionText }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(errorData.detail || 'Failed to get answer');
      }

      const data = await response.json();
      
      const assistantMessage = { 
        type: 'assistant', 
        content: data.answer,
        sources: data.sources || []
      };
      
      setMessages(prev => [...prev, assistantMessage]);
    } catch (err) {
      console.error('Query error:', err);
      setError(`Error: ${err.message}`);
      setMessages(prev => [...prev, {
        type: 'system',
        content: `❌ Error: ${err.message}`
      }]);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleSummarize = async (docId) => {
    setIsProcessing(true);
    try {
      const response = await fetch(`${API_URL}/api/summarize`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ doc_id: docId }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(errorData.detail || 'Failed to generate summary');
      }

      const data = await response.json();
      
      setMessages(prev => [...prev, {
        type: 'system',
        content: `📄 Summary for ${data.doc_id}:\n\n${data.summary}\n\nKey Points:\n${data.key_points.map(p => `• ${p}`).join('\n')}`
      }]);
    } catch (err) {
      console.error('Summarize error:', err);
      setError(`Error generating summary: ${err.message}`);
    } finally {
      setIsProcessing(false);
    }
  };

  const removeDocument = async (docId) => {
    try {
      const response = await fetch(`${API_URL}/api/documents/${docId}`, {
        method: 'DELETE',
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(errorData.detail || 'Failed to delete document');
      }

      await loadDocuments();
      setMessages(prev => [...prev, {
        type: 'system',
        content: 'Document removed from knowledge base'
      }]);
    } catch (err) {
      console.error('Delete error:', err);
      setError(`Error deleting document: ${err.message}`);
    }
  };

  const clearAll = async () => {
    try {
      // Delete all documents
      const deletePromises = documents.map(doc => 
        fetch(`${API_URL}/api/documents/${doc.doc_id}`, {
          method: 'DELETE',
        })
      );
      
      await Promise.allSettled(deletePromises);
      await loadDocuments();
      setMessages([]);
      setQuestion('');
      setError('');
      setMessages(prev => [...prev, {
        type: 'system',
        content: 'All documents cleared'
      }]);
    } catch (err) {
      console.error('Clear error:', err);
      setError(`Error clearing documents: ${err.message}`);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 p-6">
      <div className="max-w-7xl mx-auto">
        <div className="bg-white rounded-2xl shadow-xl overflow-hidden">
          {/* Header */}
          <div className="bg-gradient-to-r from-blue-600 to-indigo-600 p-6 text-white">
            <h1 className="text-3xl font-bold flex items-center gap-3">
              <FileText className="w-8 h-8" />
              RAG System - Multilingual Document Q&A
            </h1>
            <p className="mt-2 text-blue-100">Upload documents in any language and get answers in English</p>
            {!pdfJsLoaded && (
              <p className="mt-2 text-yellow-200 text-sm">⏳ Loading PDF support...</p>
            )}
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 p-6">
            {/* Left Panel - Documents */}
            <div className="lg:col-span-1 space-y-4">
              <div className="bg-gray-50 rounded-xl p-4 border-2 border-dashed border-gray-300">
                <h2 className="text-lg font-semibold mb-3 flex items-center gap-2">
                  <Upload className="w-5 h-5" />
                  Upload Documents
                </h2>
                <input
                  ref={fileInputRef}
                  type="file"
                  multiple
                  accept=".pdf,.txt,.docx,.md,.html,.csv,.rtf"
                  onChange={handleFileUpload}
                  className="hidden"
                  id="file-upload"
                  disabled={isProcessing}
                />
                <label
                  htmlFor="file-upload"
                  className={`block w-full py-3 px-4 ${isProcessing ? 'bg-gray-400' : 'bg-blue-600 hover:bg-blue-700'} text-white rounded-lg cursor-pointer text-center font-medium transition-colors`}
                >
                  {isProcessing ? 'Processing...' : 'Choose Files'}
                </label>
                <p className="text-xs text-gray-500 mt-2 text-center">
                  Supports PDF, TXT, DOCX, MD, HTML, CSV, RTF
                </p>
                <p className="text-xs text-blue-600 mt-1 text-center font-medium">
                  🌐 Any language supported
                </p>
              </div>

              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <h3 className="font-semibold text-gray-700">
                    Documents ({documents.length})
                  </h3>
                  {documents.length > 0 && (
                    <button
                      onClick={clearAll}
                      className="text-xs text-red-600 hover:text-red-800 flex items-center gap-1"
                    >
                      <Trash2 className="w-3 h-3" />
                      Clear All
                    </button>
                  )}
                </div>
                
                <div className="space-y-2 max-h-96 overflow-y-auto">
                  {documents.map(doc => (
                    <div
                      key={doc.doc_id}
                      className="bg-white p-3 rounded-lg border border-gray-200 hover:shadow-md transition-shadow"
                    >
                      <div className="flex items-start justify-between gap-2">
                        <div className="flex-1 min-w-0">
                          <p className="text-sm font-medium text-gray-800 truncate">
                            {doc.filename}
                          </p>
                          <p className="text-xs text-gray-500">
                            {doc.chunks} chunks • {(doc.text_length / 1024).toFixed(1)} KB
                          </p>
                          <p className="text-xs text-gray-400 mt-1">
                            {doc.file_type} • {new Date(doc.upload_date).toLocaleDateString()}
                          </p>
                        </div>
                        <div className="flex gap-1">
                          {doc.has_summary && (
                            <button
                              onClick={() => handleSummarize(doc.doc_id)}
                              className="text-blue-500 hover:text-blue-700 p-1"
                              title="Regenerate summary"
                            >
                              <RefreshCw className="w-4 h-4" />
                            </button>
                          )}
                          {!doc.has_summary && (
                            <button
                              onClick={() => handleSummarize(doc.doc_id)}
                              className="text-gray-400 hover:text-blue-500 p-1"
                              title="Generate summary"
                            >
                              <FileText className="w-4 h-4" />
                            </button>
                          )}
                          <button
                            onClick={() => removeDocument(doc.doc_id)}
                            className="text-red-500 hover:text-red-700 p-1"
                          >
                            <Trash2 className="w-4 h-4" />
                          </button>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Right Panel - Chat */}
            <div className="lg:col-span-2 flex flex-col h-[600px]">
              <div className="flex-1 bg-gray-50 rounded-xl p-4 overflow-y-auto mb-4 space-y-4">
                {messages.length === 0 ? (
                  <div className="flex items-center justify-center h-full text-gray-400">
                    <div className="text-center">
                      <FileText className="w-16 h-16 mx-auto mb-4 opacity-50" />
                      <p className="text-lg font-semibold mb-2">Upload documents and start asking questions!</p>
                      <p className="text-sm">Documents in any language • Answers in English</p>
                    </div>
                  </div>
                ) : (
                  messages.map((msg, idx) => (
                    <div
                      key={idx}
                      className={`flex ${msg.type === 'user' ? 'justify-end' : 'justify-start'}`}
                    >
                      <div
                        className={`max-w-[80%] rounded-xl p-4 ${
                          msg.type === 'user'
                            ? 'bg-blue-600 text-white'
                            : msg.type === 'system'
                            ? 'bg-gray-200 text-gray-700 text-sm'
                            : 'bg-white border border-gray-200'
                        }`}
                      >
                        <p className="whitespace-pre-wrap">{msg.content}</p>
                        {msg.sources && msg.sources.length > 0 && (
                          <div className="mt-3 pt-3 border-t border-gray-200">
                            <p className="text-xs text-gray-500 font-medium mb-1">Sources:</p>
                            <div className="space-y-1">
                              {msg.sources.map((source, i) => (
                                <div key={i} className="text-xs bg-blue-50 text-blue-700 px-2 py-1 rounded">
                                  <p className="font-medium">{source.filename}</p>
                                  {source.content && (
                                    <p className="text-gray-600 mt-1 line-clamp-2">{source.content}</p>
                                  )}
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  ))
                )}
                <div ref={messagesEndRef} />
              </div>

              {error && (
                <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg flex items-center gap-2 text-red-700">
                  <AlertCircle className="w-5 h-5 flex-shrink-0" />
                  <p className="text-sm">{error}</p>
                </div>
              )}

              <div className="flex gap-2">
                <input
                  type="text"
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && !isProcessing && handleAskQuestion()}
                  placeholder="Ask a question in English about your documents..."
                  className="flex-1 px-4 py-3 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500"
                  disabled={isProcessing}
                />
                <button
                  onClick={handleAskQuestion}
                  disabled={isProcessing || !question.trim() || documents.length === 0}
                  className="px-6 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-300 text-white rounded-xl font-medium transition-colors flex items-center gap-2"
                >
                  {isProcessing ? (
                    <Loader2 className="w-5 h-5 animate-spin" />
                  ) : (
                    <Send className="w-5 h-5" />
                  )}
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default RAGSystem;