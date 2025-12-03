import React, { useState, useEffect } from 'react'
import './DonorJourney.css'

const STORAGE_KEY = 'donor_journey_data'

function DonorJourney() {
  const [donations, setDonations] = useState([])
  const [donorInfo, setDonorInfo] = useState({ name: '', email: '', location: '' })
  const [loading, setLoading] = useState(false)
  const [profile, setProfile] = useState(null)
  const [recommendations, setRecommendations] = useState([])

  // Load data from localStorage on mount
  useEffect(() => {
    const saved = localStorage.getItem(STORAGE_KEY)
    if (saved) {
      try {
        const parsed = JSON.parse(saved)
        if (parsed.profile) setProfile(parsed.profile)
        if (parsed.recommendations) setRecommendations(parsed.recommendations)
        if (parsed.donorInfo) setDonorInfo(parsed.donorInfo)
        if (parsed.donations) setDonations(parsed.donations)
        console.log('Loaded saved donor journey data')
      } catch (e) {
        console.error('Error loading saved data:', e)
      }
    }
  }, [])

  // Save data to localStorage whenever it changes
  useEffect(() => {
    if (profile || recommendations.length > 0) {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({
        profile,
        recommendations,
        donorInfo,
        donations
      }))
    }
  }, [profile, recommendations, donorInfo, donations])

  const handleClear = () => {
    setProfile(null)
    setRecommendations([])
    setDonations([])
    setDonorInfo({ name: '', email: '', location: '' })
    localStorage.removeItem(STORAGE_KEY)
  }

  const handleAddDonation = () => {
    setDonations([...donations, { amount: '', category: '', description: '' }])
  }

  const handleDonationChange = (index, field, value) => {
    const updated = [...donations]
    updated[index][field] = value
    setDonations(updated)
  }

  const handleBuildProfile = async () => {
    console.log('Build Giving Identity button clicked!', { donorInfo, donations })
    
    if (donations.length === 0) {
      alert('Please add at least one donation before building your profile.')
      return
    }
    
    setLoading(true)
    setProfile(null)
    setRecommendations([])
    
    try {
      console.log('Starting profile build request...')
      
      const requestData = {
        donor_info: donorInfo,
        donations: donations.filter(d => d.amount && d.category),
      }
      
      console.log('Request data:', requestData)
      
      // Use relative URL (Vite proxy will forward to backend)
      const response = await fetch('/api/donor-journey', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestData),
      })
      
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
      console.log('Success! Profile data received:', data)
      setProfile(data.profile)
      setRecommendations(data.recommendations || [])
      // Data will be saved to localStorage via useEffect
    } catch (err) {
      console.error('Error building profile:', err)
      alert(`Error: ${err.message || 'Failed to connect to server. Make sure the backend is running on port 8000.'}`)
    } finally {
      setLoading(false)
      console.log('Profile build request completed')
    }
  }

  return (
    <div className="donor-journey">
      <div className="journey-header">
        <h2>Donor Journey Simulation</h2>
        <p>Enter past donations → Watch agents build your giving identity in real-time</p>
      </div>

      <div className="journey-form">
        <div className="form-section">
          <h3>Your Information</h3>
          <div className="form-row">
            <input
              type="text"
              placeholder="Name"
              value={donorInfo.name}
              onChange={(e) => setDonorInfo({ ...donorInfo, name: e.target.value })}
            />
            <input
              type="email"
              placeholder="Email"
              value={donorInfo.email}
              onChange={(e) => setDonorInfo({ ...donorInfo, email: e.target.value })}
            />
            <input
              type="text"
              placeholder="Location"
              value={donorInfo.location}
              onChange={(e) => setDonorInfo({ ...donorInfo, location: e.target.value })}
            />
          </div>
        </div>

        <div className="form-section">
          <div className="section-header">
            <h3>Donation History</h3>
            <button onClick={handleAddDonation} className="add-button">
              + Add Donation
            </button>
          </div>
          {donations.map((donation, index) => (
            <div key={index} className="donation-row">
              <input
                type="number"
                placeholder="Amount ($)"
                value={donation.amount}
                onChange={(e) => handleDonationChange(index, 'amount', e.target.value)}
              />
              <input
                type="text"
                placeholder="Category"
                value={donation.category}
                onChange={(e) => handleDonationChange(index, 'category', e.target.value)}
              />
              <input
                type="text"
                placeholder="Description"
                value={donation.description}
                onChange={(e) => handleDonationChange(index, 'description', e.target.value)}
              />
            </div>
          ))}
        </div>

        <button
          type="button"
          onClick={(e) => {
            console.log('Build button clicked!', e)
            e.preventDefault()
            e.stopPropagation()
            handleBuildProfile()
          }}
          disabled={loading || donations.length === 0}
          className="build-button"
          style={{ 
            cursor: (loading || donations.length === 0) ? 'not-allowed' : 'pointer',
            opacity: (loading || donations.length === 0) ? 0.6 : 1
          }}
        >
          {loading ? '⏳ Building Profile...' : '🔍 Build Giving Identity'}
        </button>
        
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
            <p style={{ fontSize: '1.1rem', fontWeight: '600' }}>Building your giving identity...</p>
            <p style={{ fontSize: '0.9rem', marginTop: '0.5rem', color: '#999' }}>
              Analyzing your donation history. This may take 30-60 seconds.
            </p>
            <p style={{ fontSize: '0.85rem', marginTop: '0.5rem', color: '#999' }}>
              Check browser console (F12) for progress updates.
            </p>
          </div>
        )}
      </div>

      {profile && (
        <div className="journey-results">
          <ProfileDisplay profile={profile} />
          {recommendations.length > 0 && (
            <RecommendationsDisplay recommendations={recommendations} />
          )}
        </div>
      )}
    </div>
  )
}

function ProfileDisplay({ profile }) {
  return (
    <div className="profile-card">
      <h3>Your Giving Identity</h3>
      <div className="profile-stats">
        <div className="stat">
          <span className="stat-label">Total Giving</span>
          <span className="stat-value">${profile.total_lifetime_giving?.toLocaleString() || 0}</span>
        </div>
        <div className="stat">
          <span className="stat-label">Average Donation</span>
          <span className="stat-value">${profile.average_donation?.toLocaleString() || 0}</span>
        </div>
        <div className="stat">
          <span className="stat-label">Donation Count</span>
          <span className="stat-value">{profile.donation_count || 0}</span>
        </div>
      </div>

      {profile.cause_affinities && profile.cause_affinities.length > 0 && (
        <div className="profile-section">
          <h4>Cause Affinities</h4>
          <div className="affinities-list">
            {profile.cause_affinities.map((aff, i) => (
              <div key={i} className="affinity-item">
                <span className="affinity-category">{aff.category}</span>
                <div className="affinity-bar">
                  <div
                    className="affinity-fill"
                    style={{ width: `${(aff.affinity_score || 0) * 100}%` }}
                  />
                </div>
                <span className="affinity-score">{Math.round((aff.affinity_score || 0) * 100)}%</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {profile.giving_motivators && profile.giving_motivators.length > 0 && (
        <div className="profile-section">
          <h4>Giving Motivators</h4>
          <ul className="motivators-list">
            {profile.giving_motivators.map((mot, i) => (
              <li key={i}>{mot}</li>
            ))}
          </ul>
        </div>
      )}

      {profile.personality_summary && (
        <div className="profile-section">
          <h4>Personality Insights</h4>
          <p className="personality-text">{profile.personality_summary}</p>
        </div>
      )}
    </div>
  )
}

function RecommendationsDisplay({ recommendations }) {
  return (
    <div className="recommendations-card">
      <h3>Personalized Campaign Recommendations</h3>
      <div className="recommendations-list">
        {recommendations.map((rec, i) => (
          <div key={i} className="recommendation-item">
            <h4>{rec.campaign_title}</h4>
            <div className="recommendation-match">
              <span>Match Score: {Math.round((rec.match_score || 0) * 100)}%</span>
            </div>
            {rec.reasons && rec.reasons.length > 0 && (
              <div className="recommendation-reasons">
                <strong>Why it matches:</strong>
                <ul>
                  {rec.reasons.map((reason, j) => (
                    <li key={j}>{reason}</li>
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

export default DonorJourney

