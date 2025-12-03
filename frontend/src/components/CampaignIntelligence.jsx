import React, { useState, useEffect } from 'react'
import './CampaignIntelligence.css'

const STORAGE_KEY = 'campaign_intelligence_results'

function CampaignIntelligence() {
  const [url, setUrl] = useState('')
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState(null)
  const [error, setError] = useState(null)

  // Load results from localStorage on mount
  useEffect(() => {
    const saved = localStorage.getItem(STORAGE_KEY)
    if (saved) {
      try {
        const parsed = JSON.parse(saved)
        setResults(parsed)
        console.log('Loaded saved campaign intelligence results')
      } catch (e) {
        console.error('Error loading saved results:', e)
      }
    }
  }, [])

  // Save results to localStorage whenever they change
  useEffect(() => {
    if (results) {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(results))
    }
  }, [results])

  const handleClear = () => {
    setResults(null)
    setError(null)
    localStorage.removeItem(STORAGE_KEY)
  }

  const handleAnalyze = async () => {
    console.log('Analyze button clicked!', { url })
    
    setLoading(true)
    setError(null)
    // Don't clear results immediately - keep previous results visible until new ones arrive

    try {
      console.log('Starting fetch request...')
      
      // Use relative URL (Vite proxy will forward to backend)
      const apiUrl = '/api/campaign-intelligence'
      const requestBody = { url: url.trim() || "" }
      
      console.log('Fetching:', apiUrl, requestBody)
      
      // Add timeout to prevent hanging (match backend timeout of 4 minutes + buffer)
      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), 250000) // 4.2 minute timeout (250 seconds)
      
      const response = await fetch(apiUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
        signal: controller.signal,
      })
      
      clearTimeout(timeoutId)

      console.log('Response received:', response.status, response.statusText)

      // Read response body once
      const responseText = await response.text()
      
      if (!response.ok) {
        let errorMessage = `Server error: ${response.status} ${response.statusText}`
        try {
          const errorData = JSON.parse(responseText)
          errorMessage = errorData.detail || errorMessage
          console.error('Server error data:', errorData)
        } catch {
          // Not JSON, use text as error message
          errorMessage = responseText || errorMessage
          console.error('Server error text:', responseText)
        }
        throw new Error(errorMessage)
      }

      // Parse JSON response
      let data
      try {
        data = JSON.parse(responseText)
      } catch (e) {
        throw new Error('Invalid JSON response from server')
      }
      console.log('Success! Data received:', data)
      
      // Handle partial results with errors gracefully
      if (data.error) {
        console.warn('Response contains error:', data.error)
        // Still show the results, but also show the error message
        setResults(data)
        setError(`Warning: ${data.error}`)
      } else {
        setResults(data)
        setError(null) // Clear any previous errors on success
      }
    } catch (err) {
      console.error('Fetch error:', err)
      // Set error but don't clear results - keep previous results visible
      if (err.name === 'AbortError') {
        setError('Request timed out. The analysis is taking longer than expected. Please try again or check if the backend is still running.')
      } else if (err.message.includes('Failed to fetch') || err.message.includes('NetworkError') || err.name === 'TypeError') {
        setError('Cannot connect to server. Please make sure the backend is running on port 8000. Run: uvicorn src.api.ag_ui_server:app --reload --port 8000')
      } else {
        setError(err.message || 'Failed to analyze campaign. Please try again.')
      }
      // If we have previous results, keep them; otherwise set results to null
      if (!results) {
        setResults(null)
      }
    } finally {
      setLoading(false)
      console.log('Request completed')
    }
  }

  return (
    <div className="campaign-intelligence">
      <div className="intelligence-header">
        <h2>Live Campaign Intelligence</h2>
        <p>Paste any GoFundMe URL → Get instant AI analysis</p>
      </div>

      <div className="intelligence-input">
        <input
          type="text"
          placeholder="Enter GoFundMe URL (or leave empty for demo)..."
          value={url}
          onChange={(e) => {
            console.log('URL changed:', e.target.value)
            setUrl(e.target.value)
          }}
          onKeyPress={(e) => {
            if (e.key === 'Enter') {
              console.log('Enter key pressed')
              handleAnalyze()
            }
          }}
        />
        <button 
          type="button"
          onClick={(e) => {
            console.log('Button clicked!', e)
            e.preventDefault()
            e.stopPropagation()
            handleAnalyze()
          }} 
          disabled={loading}
          style={{ 
            cursor: loading ? 'wait' : 'pointer',
            opacity: loading ? 0.6 : 1
          }}
        >
          {loading ? '⏳ Analyzing...' : '🔍 Analyze Campaign'}
        </button>
      </div>
      
      {loading && (
        <div style={{ 
          textAlign: 'center', 
          padding: '2rem', 
          color: '#666',
          background: '#f8f8f8',
          borderRadius: '8px',
          marginTop: '1rem'
        }}>
          <div style={{ fontSize: '2rem', marginBottom: '1rem' }}>⏳</div>
          <p style={{ fontSize: '1.1rem', fontWeight: '600' }}>Analyzing campaign...</p>
          <p style={{ fontSize: '0.9rem', marginTop: '0.5rem', color: '#999' }}>
            This may take 30-60 seconds. Check browser console (F12) for progress.
          </p>
        </div>
      )}

      {error && <div className="error-message">{error}</div>}

      {results && (
        <div className="intelligence-results">
          {(() => {
            try {
              return (
                <>
                  {results.extracted_campaign_summary && (
                    <ExtractedCampaignSummary summary={results.extracted_campaign_summary} />
                  )}
                  {results.tone_analysis && (
                    <ToneAnalysis tone={results.tone_analysis} />
                  )}
                  {results.quality_score && (
                    <QualityScore score={results.quality_score} />
                  )}
                  {results.success_prediction && (
                    <SuccessPrediction prediction={results.success_prediction} />
                  )}
                  {results.similar_campaigns && results.similar_campaigns.length > 0 && (
                    <SimilarCampaigns campaigns={results.similar_campaigns} />
                  )}
                  {results.messaging_variants && Object.keys(results.messaging_variants).length > 0 && (
                    <MessagingVariants variants={results.messaging_variants} />
                  )}
                  {results.ab_testing && (
                    <ABTesting abTesting={results.ab_testing} />
                  )}
                </>
              )
            } catch (err) {
              console.error('Error rendering results:', err)
              return (
                <div className="result-card" style={{ background: '#ffebee', borderLeft: '4px solid #f44336' }}>
                  <h3>Error Displaying Results</h3>
                  <p>There was an error rendering some results. Please try refreshing the page.</p>
                  <p style={{ fontSize: '0.9rem', color: '#666' }}>Error: {err.message}</p>
                </div>
              )
            }
          })()}
        </div>
      )}
    </div>
  )
}

function ExtractedCampaignSummary({ summary }) {
  return (
    <div className="result-card" style={{ 
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      color: 'white',
      marginBottom: '2rem'
    }}>
      <h3 style={{ marginTop: 0, color: 'white' }}>📋 Extracted Campaign Summary</h3>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginTop: '1rem' }}>
        <div>
          <strong>Title:</strong>
          <p style={{ margin: '0.5rem 0', fontSize: '1.1rem' }}>{summary.title}</p>
        </div>
        <div>
          <strong>Category:</strong>
          <p style={{ margin: '0.5rem 0', textTransform: 'capitalize' }}>{summary.category}</p>
        </div>
        <div>
          <strong>Description:</strong>
          <p style={{ margin: '0.5rem 0', opacity: 0.9 }}>{summary.description || 'No description available'}</p>
        </div>
        <div>
          <strong>Organizer:</strong>
          <p style={{ margin: '0.5rem 0' }}>{summary.organizer_name || 'Not specified'}</p>
        </div>
        <div>
          <strong>Location:</strong>
          <p style={{ margin: '0.5rem 0' }}>{summary.location || 'Not specified'}</p>
        </div>
        <div>
          <strong>URL:</strong>
          <p style={{ margin: '0.5rem 0', wordBreak: 'break-all' }}>
            <a href={summary.url} target="_blank" rel="noopener noreferrer" style={{ color: 'white', textDecoration: 'underline' }}>
              {summary.url}
            </a>
          </p>
        </div>
      </div>
      <div style={{ 
        display: 'grid', 
        gridTemplateColumns: 'repeat(3, 1fr)', 
        gap: '1rem', 
        marginTop: '1.5rem',
        paddingTop: '1.5rem',
        borderTop: '1px solid rgba(255,255,255,0.3)'
      }}>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>${summary.goal_amount.toLocaleString()}</div>
          <div style={{ fontSize: '0.9rem', opacity: 0.9 }}>Goal Amount</div>
        </div>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>${summary.raised_amount.toLocaleString()}</div>
          <div style={{ fontSize: '0.9rem', opacity: 0.9 }}>Raised Amount</div>
        </div>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>{summary.donor_count}</div>
          <div style={{ fontSize: '0.9rem', opacity: 0.9 }}>Donors</div>
        </div>
      </div>
      {summary.goal_amount > 0 && (
        <div style={{ marginTop: '1rem', paddingTop: '1rem', borderTop: '1px solid rgba(255,255,255,0.3)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
            <span>Progress:</span>
            <span style={{ fontWeight: 'bold' }}>{summary.progress_percentage.toFixed(1)}%</span>
          </div>
          <div style={{ 
            width: '100%', 
            height: '8px', 
            background: 'rgba(255,255,255,0.3)', 
            borderRadius: '4px',
            overflow: 'hidden'
          }}>
            <div style={{ 
              width: `${Math.min(summary.progress_percentage, 100)}%`, 
              height: '100%', 
              background: 'white',
              transition: 'width 0.3s'
            }}></div>
          </div>
        </div>
      )}
    </div>
  )
}

function ToneAnalysis({ tone }) {
  if (!tone) {
    return null
  }
  
  return (
    <div className="result-card" style={{ borderLeft: '4px solid #4caf50' }}>
      <h3>💬 Tone Analysis</h3>
      <div className="score-overall">
        <span className="score-value">{Math.round((tone.overall_tone_score || 0) * 100)}%</span>
        <div className="score-bar">
          <div
            className="score-fill"
            style={{ width: `${(tone.overall_tone_score || 0) * 100}%`, background: '#4caf50' }}
          />
        </div>
      </div>
      
      <div className="dimension-scores" style={{ marginTop: '1.5rem' }}>
        <div className="dimension-item">
          <span className="dimension-name">Empathy</span>
          <div className="dimension-bar">
            <div className="dimension-fill" style={{ width: `${(tone.empathy_score || 0) * 100}%` }} />
          </div>
          <span className="dimension-value">{Math.round((tone.empathy_score || 0) * 100)}%</span>
        </div>
        <div className="dimension-item">
          <span className="dimension-name">Authenticity</span>
          <div className="dimension-bar">
            <div className="dimension-fill" style={{ width: `${(tone.authenticity_score || 0) * 100}%` }} />
          </div>
          <span className="dimension-value">{Math.round((tone.authenticity_score || 0) * 100)}%</span>
        </div>
        <div className="dimension-item">
          <span className="dimension-name">Clarity</span>
          <div className="dimension-bar">
            <div className="dimension-fill" style={{ width: `${(tone.clarity_score || 0) * 100}%` }} />
          </div>
          <span className="dimension-value">{Math.round((tone.clarity_score || 0) * 100)}%</span>
        </div>
        <div className="dimension-item">
          <span className="dimension-name">Urgency Appropriateness</span>
          <div className="dimension-bar">
            <div className="dimension-fill" style={{ width: `${(tone.urgency_appropriateness || 0) * 100}%` }} />
          </div>
          <span className="dimension-value">{Math.round((tone.urgency_appropriateness || 0) * 100)}%</span>
        </div>
        <div className="dimension-item">
          <span className="dimension-name">Respectful Language</span>
          <div className="dimension-bar">
            <div className="dimension-fill" style={{ width: `${(tone.respectful_language_score || 0) * 100}%` }} />
          </div>
          <span className="dimension-value">{Math.round((tone.respectful_language_score || 0) * 100)}%</span>
        </div>
      </div>
      
      {tone.tone_strengths && tone.tone_strengths.length > 0 && (
        <div style={{ marginTop: '1.5rem', padding: '1rem', background: '#e8f5e9', borderRadius: '6px' }}>
          <h4 style={{ marginTop: 0, color: '#2e7d32' }}>✓ Tone Strengths</h4>
          <ul style={{ marginBottom: 0 }}>
            {tone.tone_strengths.map((strength, i) => (
              <li key={i}>{strength}</li>
            ))}
          </ul>
        </div>
      )}
      
      {tone.tone_issues && tone.tone_issues.length > 0 && (
        <div style={{ marginTop: '1rem', padding: '1rem', background: '#fff3e0', borderRadius: '6px' }}>
          <h4 style={{ marginTop: 0, color: '#e65100' }}>⚠ Tone Issues</h4>
          <ul style={{ marginBottom: 0 }}>
            {tone.tone_issues.map((issue, i) => (
              <li key={i}>{issue}</li>
            ))}
          </ul>
        </div>
      )}
      
      {tone.sensitive_phrases && tone.sensitive_phrases.length > 0 && (
        <div style={{ marginTop: '1rem', padding: '1rem', background: '#ffebee', borderRadius: '6px' }}>
          <h4 style={{ marginTop: 0, color: '#c62828' }}>⚠ Sensitive Phrases Detected</h4>
          <ul style={{ marginBottom: 0 }}>
            {tone.sensitive_phrases.map((phrase, i) => (
              <li key={i}><code>{phrase}</code></li>
            ))}
          </ul>
        </div>
      )}
      
      {tone.recommended_changes && tone.recommended_changes.length > 0 && (
        <div style={{ marginTop: '1rem', padding: '1rem', background: '#e3f2fd', borderRadius: '6px' }}>
          <h4 style={{ marginTop: 0, color: '#1565c0' }}>💡 Recommended Changes</h4>
          <ul style={{ marginBottom: 0 }}>
            {tone.recommended_changes.map((change, i) => (
              <li key={i}>{change}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}

function QualityScore({ score }) {
  return (
    <div className="result-card">
      <h3>Campaign Quality Score</h3>
      <div className="score-overall">
        <span className="score-value">{Math.round((score.overall_score || 0) * 100)}%</span>
        <div className="score-bar">
          <div
            className="score-fill"
            style={{ width: `${(score.overall_score || 0) * 100}%` }}
          />
        </div>
      </div>

      {score.dimension_scores && Object.keys(score.dimension_scores).length > 0 && (
        <div className="dimension-scores">
          {Object.entries(score.dimension_scores).map(([dim, val]) => (
            <div key={dim} className="dimension-item">
              <span className="dimension-name">{dim.replace('_', ' ')}</span>
              <div className="dimension-bar">
                <div
                  className="dimension-fill"
                  style={{ width: `${(val || 0) * 100}%` }}
                />
              </div>
              <span className="dimension-value">{Math.round((val || 0) * 100)}%</span>
            </div>
          ))}
        </div>
      )}

      {score.priority_improvements && score.priority_improvements.length > 0 && (
        <div className="improvements">
          <h4>Priority Improvements</h4>
          <ul>
            {score.priority_improvements.map((imp, i) => (
              <li key={i}>{imp}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}

function SuccessPrediction({ prediction }) {
  if (!prediction) {
    return null
  }
  return (
    <div className="result-card">
      <h3>Success Prediction</h3>
      <div className="prediction-main">
        <div className="prediction-probability">
          <span className="prob-value">{Math.round((prediction.success_probability || 0) * 100)}%</span>
          <span className="prob-label">Success Probability</span>
        </div>
        <div className="prediction-details">
          <div className="prediction-item">
            <span className="pred-label">Predicted Amount:</span>
            <span className="pred-value">${(prediction.predicted_amount || 0).toLocaleString()}</span>
          </div>
          <div className="prediction-item">
            <span className="pred-label">Predicted Donors:</span>
            <span className="pred-value">{prediction.predicted_donors || 0}</span>
          </div>
          <div className="prediction-item">
            <span className="pred-label">Confidence:</span>
            <span className={`pred-value confidence-${prediction.confidence_level || 'medium'}`}>
              {prediction.confidence_level || 'medium'}
            </span>
          </div>
        </div>
      </div>

      <div className="prediction-factors">
        {prediction.key_factors && prediction.key_factors.length > 0 && (
          <div className="factors-section">
            <h4>Key Success Factors</h4>
            <ul>
              {prediction.key_factors.map((factor, i) => (
                <li key={i}>✓ {factor}</li>
              ))}
            </ul>
          </div>
        )}
        {prediction.risk_factors && prediction.risk_factors.length > 0 && (
          <div className="factors-section">
            <h4>Risk Factors</h4>
            <ul>
              {prediction.risk_factors.map((risk, i) => (
                <li key={i}>⚠ {risk}</li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  )
}

function SimilarCampaigns({ campaigns }) {
  if (!campaigns || campaigns.length === 0) {
    return null
  }
  
  return (
    <div className="result-card">
      <h3>Similar Successful Campaigns</h3>
      <div className="similar-campaigns-list">
        {campaigns.map((campaign, i) => (
          <div key={i} className="similar-campaign">
            <h4>{campaign.title || `Campaign ${i + 1}`}</h4>
            {campaign.success_metrics && (
              <div className="campaign-metrics">
                {campaign.success_metrics.raised !== undefined && (
                  <span>Raised: ${campaign.success_metrics.raised.toLocaleString()}</span>
                )}
                {campaign.success_metrics.donors !== undefined && (
                  <span>Donors: {campaign.success_metrics.donors}</span>
                )}
                {campaign.similarity_score !== undefined && (
                  <span>Similarity: {Math.round(campaign.similarity_score * 100)}%</span>
                )}
              </div>
            )}
            {campaign.what_made_it_work && campaign.what_made_it_work.length > 0 && (
              <div className="what-worked">
                <h5>What Made It Work:</h5>
                <ul>
                  {campaign.what_made_it_work.map((item, j) => (
                    <li key={j}>{item}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}

function MessagingVariants({ variants }) {
  if (!variants || Object.keys(variants).length === 0) {
    return null
  }
  
  return (
    <div className="result-card">
      <h3>📝 Messaging Variants</h3>
      <div className="strategy-messaging">
        {Object.entries(variants).map(([channel, message]) => (
          <div key={channel} className="message-variant" style={{ marginBottom: '1rem', padding: '1rem', background: '#f5f5f5', borderRadius: '4px' }}>
            <strong style={{ display: 'block', marginBottom: '0.5rem' }}>{channel}:</strong>
            <p style={{ margin: 0 }}>{message}</p>
          </div>
        ))}
      </div>
    </div>
  )
}

function ABTesting({ abTesting }) {
  if (!abTesting) {
    return null
  }
  
  const variants = abTesting.variants || []
  const strategy = abTesting.testing_strategy || 'No strategy available'
  
  return (
    <div className="result-card">
      <h3>🧪 A/B Testing Plan</h3>
      
      {variants.length > 0 ? (
        <>
          <div className="ab-variants" style={{ marginBottom: '1.5rem' }}>
            <h4>Message Variants ({variants.length})</h4>
            {variants.map((variant, i) => (
              <div key={i} className="variant-item" style={{ 
                marginBottom: '1rem', 
                padding: '1rem', 
                background: '#f9f9f9', 
                borderRadius: '4px',
                borderLeft: '4px solid #2196F3'
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                  <strong>{variant.variant_name || `Variant ${i + 1}`}</strong>
                  <span style={{ 
                    padding: '0.25rem 0.5rem', 
                    background: variant.testing_priority === 'high' ? '#ff9800' : '#4caf50',
                    color: 'white',
                    borderRadius: '4px',
                    fontSize: '0.85rem'
                  }}>
                    {variant.testing_priority || 'medium'}
                  </span>
                </div>
                <div style={{ marginBottom: '0.5rem' }}>
                  <strong>Channel:</strong> {variant.channel || 'General'}
                </div>
                <div style={{ marginBottom: '0.5rem' }}>
                  <strong>Message:</strong>
                  <p style={{ margin: '0.5rem 0', whiteSpace: 'pre-wrap' }}>{variant.message || 'No message provided'}</p>
                </div>
                {variant.hypothesis && (
                  <div style={{ marginBottom: '0.5rem', fontSize: '0.9rem', color: '#666' }}>
                    <strong>Hypothesis:</strong> {variant.hypothesis}
                  </div>
                )}
                {variant.target_audience && (
                  <div style={{ fontSize: '0.9rem', color: '#666' }}>
                    <strong>Target Audience:</strong> {variant.target_audience}
                  </div>
                )}
              </div>
            ))}
          </div>
          
          {abTesting.recommended_tests && abTesting.recommended_tests.length > 0 && (
            <div className="recommended-tests" style={{ marginBottom: '1.5rem' }}>
              <h4>Recommended Tests</h4>
              <ul>
                {abTesting.recommended_tests.map((test, i) => (
                  <li key={i}>{test.test_name || JSON.stringify(test)}</li>
                ))}
              </ul>
            </div>
          )}
        </>
      ) : (
        <div style={{ padding: '1rem', background: '#fff3cd', borderRadius: '4px', color: '#856404' }}>
          <p style={{ margin: 0 }}>{strategy}</p>
        </div>
      )}
      
      {strategy && variants.length > 0 && (
        <div className="testing-strategy" style={{ marginTop: '1.5rem', padding: '1rem', background: '#e3f2fd', borderRadius: '4px' }}>
          <h4>Testing Strategy</h4>
          <p style={{ margin: 0 }}>{strategy}</p>
        </div>
      )}
    </div>
  )
}

export default CampaignIntelligence

