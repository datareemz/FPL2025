output "bucket_name" {
  value       = google_storage_bucket.fpl_bucket.name
  description = "The name of the Cloud Storage bucket."
}

output "bucket_url" {
  value       = google_storage_bucket.fpl_bucket.url
  description = "The URL of the Cloud Storage bucket."
}

output "bucket_self_link" {
  value       = google_storage_bucket.fpl_bucket.self_link
  description = "The self link of the Cloud Storage bucket."
}