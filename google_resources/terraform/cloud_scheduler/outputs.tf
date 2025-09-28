output "scheduler_service_account_email" {
  value       = google_service_account.scheduler_invoker.email
  description = "The email of the service account used by Cloud Scheduler."
}

output "scheduler_job_name" {
  value       = google_cloud_scheduler_job.schedule.name
  description = "The name of the Cloud Scheduler job."
}