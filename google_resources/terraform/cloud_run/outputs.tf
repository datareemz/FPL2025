output "job_name" {
  description = "The name of the Cloud Run job"
  value       = google_cloud_run_v2_job.python_job.name
}

output "job_id" {
  description = "The full ID of the Cloud Run job"
  value       = google_cloud_run_v2_job.python_job.id
}

output "job_location" {
  description = "The location of the Cloud Run job"
  value       = google_cloud_run_v2_job.python_job.location
}